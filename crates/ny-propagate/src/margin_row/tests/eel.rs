// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-errlane (margin-row throughput plan B3) soundness and identity tests.
//!
//! * `eel_dominates_percol_every_op` — the per-op invariant: after EVERY op
//!   (and after each ReLU sub-step: gate, exception update, shared beta charge,
//!   per-column beta charge; the stacked arm's three likewise) the envelope
//!   lane's materialized error is `>=` the per-column lane's, entry by entry,
//!   bit-for-bit, and the coefficient lane is bit-identical.
//! * `eel_exact_rational_falsifier_encloses_sampled_forwards` — the published
//!   root bound under each envelope lane still lower-bounds the EXACT
//!   (`BigRational`) margin at every sampled point, including POINT boxes
//!   where the relaxation is exact and only the certified error lane stands
//!   between the bound and the truth.
//! * `eel_adversarial_bias_rounding_oracle_holds_under_envelopes` — the
//!   calibrated running-bias oracle (which caught a real false UNSAT in the
//!   `eb` lane) re-run with the envelope lanes carrying `eb`.
//! * `eel_off_is_bit_identical` — `off` (and every arm that does not engage:
//!   `envelope` below its width floor, and `audit`, which publishes the
//!   per-column lane) is byte-identical per pass AND per lane verdict.

use super::*;

use crate::margin_row::engine::{BetaSplit, Collect, PassOut, PcBetaCol, Seed};
use crate::margin_row::errlane::{self, trace, ErrLane, PassLane, ENV_MIN_COLS};
use num_rational::BigRational;

/// The two envelope lanes, as engines see them.
const ENVELOPES: [ErrLane; 2] = [ErrLane::Envelope, ErrLane::PerChannel];

fn fixtures() -> Vec<(&'static str, TwinSpec, Vec<f64>, Vec<f64>)> {
    let mut out = Vec::new();
    let mut rng = StdRng::seed_from_u64(0xE1_E1);
    out.push((
        "tiny_0.6",
        tiny_spec(&mut rng, 0.6),
        vec![-0.3; 72],
        vec![0.3; 72],
    ));
    let mut rng = StdRng::seed_from_u64(0xE2_E2);
    out.push((
        "tiny_0.9",
        tiny_spec(&mut rng, 0.9),
        vec![-0.2; 72],
        vec![0.35; 72],
    ));
    // ConvTranspose + ChannelAffine + stride-2 Conv: the op set tiny_spec
    // lacks (affine envelope, transpose-built tap tables, channel groups that
    // change size across a conv).
    let mut rng = StdRng::seed_from_u64(0xE3_E3);
    out.push((
        "cgan_like",
        cgan_like_spec(&mut rng, 0.6),
        vec![-0.4; 9],
        vec![0.5; 9],
    ));
    out
}

/// Materialize nothing, compare everything: bit-identity of the coefficient
/// lane and `>=` of both error lanes, with a readable failure.
fn assert_pass_dominates(col: &PassOut, env: &PassOut, what: &str) {
    assert_eq!(col.a.raw_dim(), env.a.raw_dim(), "{what}: shape");
    for (i, (x, y)) in col.a.iter().zip(env.a.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: coefficient {i} moved");
    }
    for (i, (x, y)) in col.b.iter().zip(&env.b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: bias {i} moved");
    }
    let (ce, ee) = (
        col.e.as_ref().expect("outward per-column lane"),
        env.e.as_ref().expect("outward envelope lane"),
    );
    assert_eq!(ce.raw_dim(), ee.raw_dim(), "{what}: error shape");
    for ((ij, x), y) in ce.indexed_iter().zip(ee.iter()) {
        assert!(y >= x, "{what}: E_hat {y:e} < E {x:e} at {ij:?}");
    }
    for (c, (x, y)) in col.eb.iter().zip(&env.eb).enumerate() {
        assert!(y >= x, "{what}: eb_hat {y:e} < eb {x:e} at column {c}");
    }
}

/// Zip two per-op traces and assert the invariant at EVERY step. Returns
/// `(steps, strictly_looser_entries)`.
fn assert_trace_dominates(col: &[trace::Step], env: &[trace::Step], what: &str) -> (usize, usize) {
    assert_eq!(col.len(), env.len(), "{what}: trace lengths differ");
    assert!(
        !col.is_empty(),
        "{what}: empty trace (capture did not engage)"
    );
    let mut strict = 0usize;
    for (i, (c, e)) in col.iter().zip(env).enumerate() {
        assert_eq!(
            (c.op, c.tag),
            (e.op, e.tag),
            "{what}: step {i} sequence differs"
        );
        assert_eq!(c.l.raw_dim(), e.l.raw_dim(), "{what}: step {i} shape");
        for (x, y) in c.l.iter().zip(e.l.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}: step {i} ({}) coefficient lane moved",
                c.tag
            );
        }
        let (ce, ee) = (
            c.e.as_ref().expect("per-column step carries E"),
            e.e.as_ref().expect("envelope step carries E_hat"),
        );
        assert_eq!(ce.raw_dim(), ee.raw_dim(), "{what}: step {i} error shape");
        for ((ij, x), y) in ce.indexed_iter().zip(ee.iter()) {
            assert!(
                y >= x,
                "{what}: step {i} op {} ({}): E_hat {y:e} < E {x:e} at {ij:?}",
                c.op,
                c.tag
            );
            if y > x {
                strict += 1;
            }
        }
        assert_eq!(c.eb.len(), e.eb.len(), "{what}: step {i} eb width");
        for (col_i, (x, y)) in c.eb.iter().zip(&e.eb).enumerate() {
            assert!(
                y >= x,
                "{what}: step {i} ({}): eb_hat {y:e} < eb {x:e} at column {col_i}",
                c.tag
            );
        }
    }
    (col.len(), strict)
}

/// A dense deterministic seed with a certified error lane.
fn dense_seed(n_y: usize, r: usize, salt: usize) -> Seed {
    let s = Array2::from_shape_fn((n_y, r), |(j, c)| {
        ((1 + salt * 31 + j * 13 + c * 7) as f64 * 0.217).sin()
    });
    let e = Array2::from_shape_fn((n_y, r), |(j, c)| {
        (1 + (salt + j * 3 + c * 5) % 11) as f64 * 1e-16
    });
    Seed { s, e: Some(e) }
}

/// Domain gates with a split, shared beta terms and per-column beta terms,
/// plus per-(row, neuron) exceptions at every trunk layer.
fn armed_domain(root: &RootGates, r: usize) -> (DomainGates, Exceptions) {
    let unst: Vec<(usize, Vec<usize>)> = root
        .layers
        .iter()
        .enumerate()
        .map(|(li, rec)| (li, rec.unst.clone()))
        .collect();
    let splits: Vec<(usize, usize, i8)> = unst
        .iter()
        .filter(|(_, u)| !u.is_empty())
        .take(1)
        .map(|(li, _)| (*li, 0usize, 1i8))
        .collect();
    let mut dom = domain_gates(root, &splits);
    let mut exc = Exceptions::default();
    for (li, rec) in root.layers.iter().enumerate() {
        let n = rec.n;
        // Shared beta: two terms of opposite sign at two neurons.
        dom.beta.insert(
            li,
            vec![
                BetaSplit {
                    neuron: (li * 3) % n,
                    sign: 1,
                    beta: 0.375,
                },
                BetaSplit {
                    neuron: (li * 3 + 1) % n,
                    sign: -1,
                    beta: 1.25,
                },
            ],
        );
        // Per-column beta on two (distinct, in-range) columns.
        dom.beta_pc.insert(
            li,
            (0..r.min(2))
                .map(|c| PcBetaCol {
                    col: (c + 1) % r,
                    terms: vec![BetaSplit {
                        neuron: (li * 5 + c + 2) % n,
                        sign: if c % 2 == 0 { 1 } else { -1 },
                        beta: 0.5 + c as f64,
                    }],
                })
                .collect(),
        );
        // Exceptions: split-active, split-inactive and a fractional slope;
        // distinct (row, neuron) keys by construction.
        let list: Vec<Exc> = [(1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.35, 0.8, 0.02)]
            .iter()
            .enumerate()
            .filter(|(k, _)| *k < r)
            .map(|(k, &(a2, s2, c2))| Exc {
                row: k,
                neuron: (li * 7 + 2 * k + 1) % n,
                a2,
                s2,
                c2,
            })
            .collect();
        exc.by_layer.insert(li, list);
    }
    (dom, exc)
}

/// THE PER-OP INVARIANT: `Ê >= E[., c]` after every op, for both envelope
/// groupings, over margin / identity / dense seeds, both directions, with and
/// without split gates + beta + exceptions, a prefix (backward-interm) pass
/// and the domain-stacked arm. Non-vacuity is asserted, not assumed: every
/// sub-step tag must appear, exceptions and both beta kinds must MOVE the
/// coefficient lane, the envelope rows must be broadcast (the envelope really
/// ran) and strictly looser somewhere.
#[test]
fn eel_dominates_percol_every_op() {
    let mut tags_seen = std::collections::BTreeSet::new();
    let mut total_steps = 0usize;
    let mut total_strict = 0usize;
    let mut exc_moved = false;
    let mut beta_moved = false;
    let mut beta_pc_moved = false;
    for (name, spec, lo, hi) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        assert_eq!(
            root.root_precision_policy.err_lane,
            ErrLane::PerCol,
            "fixture roots must build dark"
        );
        let col = BackwardEngine::new(&net, &root);
        let (t, _, adv) = tiny_targets(&net, &lo, &hi);
        let mb = MarginBatch::new(&net, t, &adv).expect("mb");
        let (al, au) = col.y_rows(None).expect("y rows");
        let ybox = YBox::from_rows(&col, &al, &au);
        let gates = head_gates(&ybox, RoundMode::Outward);
        let margin = margin_seed(&mb, &gates, &ybox, RoundMode::Outward).seed;
        let seeds: Vec<(&str, Seed)> = vec![
            ("margin", margin),
            ("identity", col.identity_seed()),
            ("dense7", dense_seed(net.n_y, 7, 3)),
        ];
        for lane in ENVELOPES {
            let env = BackwardEngine::new(&net, &root)
                .with_err_lane(lane)
                .with_err_min_cols(1);
            for (sname, seed) in &seeds {
                let r = seed.s.ncols();
                let (dom, exc) = armed_domain(&root, r);
                for armed in [false, true] {
                    for dir in [LaneDir::Lower, LaneDir::Upper] {
                        let what = format!("{name}/{lane:?}/{sname}/armed={armed}/{dir:?}");
                        let (d, x) = if armed {
                            (Some(&dom), Some(&exc))
                        } else {
                            (None, None)
                        };
                        let (oc, tc) = trace::capture(|| {
                            col.run_collect(seed, d, dir, x, Collect::default())
                                .expect("per-column pass")
                        });
                        let (oe, te) = trace::capture(|| {
                            env.run_collect(seed, d, dir, x, Collect::default())
                                .expect("envelope pass")
                        });
                        let (steps, strict) = assert_trace_dominates(&tc, &te, &what);
                        assert_pass_dominates(&oc, &oe, &what);
                        // The envelope REALLY ran: its published E is constant
                        // along every row (a broadcast), unlike the per-column E.
                        let ee = oe.e.as_ref().expect("outward");
                        for row in ee.rows() {
                            assert!(
                                row.iter().all(|v| v.to_bits() == row[0].to_bits()),
                                "{what}: envelope row is not a broadcast — the lane did not engage"
                            );
                        }
                        if lane == ErrLane::PerChannel {
                            // ...and constant within every input CHANNEL.
                            let groups = errlane::channel_groups(&net);
                            let g0 = groups[0];
                            if g0 > 1 {
                                for (ch, block) in ee.column(0).to_vec().chunks(g0).enumerate() {
                                    assert!(
                                        block.iter().all(|v| v.to_bits() == block[0].to_bits()),
                                        "{what}: channel {ch} envelope not per-channel"
                                    );
                                }
                            }
                        }
                        // Concretized bounds can only loosen.
                        let lower = matches!(dir, LaneDir::Lower);
                        let (bc, be) = if lower {
                            (col.concretize_lower(&oc), env.concretize_lower(&oe))
                        } else {
                            (col.concretize_upper(&oc), env.concretize_upper(&oe))
                        };
                        for (k, (c, e)) in bc.iter().zip(&be).enumerate() {
                            assert!(
                                if lower { e <= c } else { e >= c },
                                "{what}: concretized bound {k} TIGHTENED under the envelope \
                                 ({e:e} vs {c:e})"
                            );
                        }
                        total_steps += steps;
                        total_strict += strict;
                        for st in &tc {
                            tags_seen.insert(st.tag);
                        }
                        if armed {
                            for w in tc.windows(2) {
                                let moved = w[0].l.iter().zip(w[1].l.iter()).any(|(a, b)| a != b);
                                match (w[0].tag, w[1].tag) {
                                    ("relu.gate", "relu.exc") => exc_moved |= moved,
                                    ("relu.exc", "relu.beta") => beta_moved |= moved,
                                    ("relu.beta", "relu.beta_pc") => beta_pc_moved |= moved,
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            // #backward-interm PREFIX pass: identity rows at a trunk relu's
            // input (disjoint column supports — the envelope's worst case).
            for rec in &root.layers {
                let n_pre = rec.n;
                let r = n_pre.min(9);
                let mut s = Array2::<f64>::zeros((n_pre, r));
                for c in 0..r {
                    s[[(c * 5) % n_pre, c]] = 1.0;
                }
                let seed = Seed { s, e: None };
                for dir in [LaneDir::Lower, LaneDir::Upper] {
                    let what = format!("{name}/{lane:?}/prefix@{}/{dir:?}", rec.op);
                    let (oc, tc) = trace::capture(|| {
                        col.run_prefix(&seed, rec.op, dir)
                            .expect("per-column prefix")
                    });
                    let (oe, te) = trace::capture(|| {
                        env.run_prefix(&seed, rec.op, dir).expect("envelope prefix")
                    });
                    let (steps, strict) = assert_trace_dominates(&tc, &te, &what);
                    assert_pass_dominates(&oc, &oe, &what);
                    total_steps += steps;
                    total_strict += strict;
                }
            }
            // DOMAIN-STACKED arm: three domains, own gates, own exceptions,
            // shared-beta terms in one block.
            let (dom_a, exc_a) = armed_domain(&root, 3);
            let mut dom_a = dom_a;
            dom_a.beta_pc.clear(); // the stacked arm refuses per-column beta
            let dom_b = domain_gates(&root, &[]);
            let exc_b = Exceptions::default();
            let dom_c = domain_gates(&root, &[]);
            let (_, exc_c) = armed_domain(&root, 2);
            let widths = [3usize, 4, 2];
            let total: usize = widths.iter().sum();
            let seed = dense_seed(net.n_y, total, 11);
            let blocks = vec![
                RowDomainGateBlock {
                    columns: 0..3,
                    gates: &dom_a,
                    exceptions: &exc_a,
                },
                RowDomainGateBlock {
                    columns: 3..7,
                    gates: &dom_b,
                    exceptions: &exc_b,
                },
                RowDomainGateBlock {
                    columns: 7..9,
                    gates: &dom_c,
                    exceptions: &exc_c,
                },
            ];
            for dir in [LaneDir::Lower, LaneDir::Upper] {
                let what = format!("{name}/{lane:?}/stacked/{dir:?}");
                let (oc, tc) = trace::capture(|| {
                    col.run_domain_stacked(&seed, &blocks, dir)
                        .expect("per-column stacked")
                });
                let (oe, te) = trace::capture(|| {
                    env.run_domain_stacked(&seed, &blocks, dir)
                        .expect("envelope stacked")
                });
                let (steps, strict) = assert_trace_dominates(&tc, &te, &what);
                assert_pass_dominates(&oc, &oe, &what);
                total_steps += steps;
                total_strict += strict;
                for st in &tc {
                    tags_seen.insert(st.tag);
                }
            }
        }
    }
    for tag in [
        "seed",
        "merge",
        "affine",
        "relu.gate",
        "relu.exc",
        "relu.beta",
        "relu.beta_pc",
        "stack.gate",
        "stack.exc",
        "stack.beta",
    ] {
        assert!(
            tags_seen.contains(tag),
            "no step tagged {tag:?} was ever checked"
        );
    }
    assert!(
        exc_moved,
        "exceptions never moved the coefficient lane (vacuous)"
    );
    assert!(
        beta_moved,
        "shared beta never moved the coefficient lane (vacuous)"
    );
    assert!(
        beta_pc_moved,
        "per-column beta never moved the coefficient lane (vacuous)"
    );
    assert!(
        total_strict > 0,
        "the envelope was never strictly looser than the per-column lane — \
         the comparison is vacuous (did the envelope run at all?)"
    );
    eprintln!(
        "[mr-errlane-test] per-op dominance: {total_steps} steps checked, \
         {total_strict} strictly-looser error entries, tags={tags_seen:?}"
    );
}

/// Exact dyadic-rational forward over the raw TwinSpec at one point: class
/// scores with NO rounding at all (every f64 is a dyadic rational; ReLU is an
/// exact comparison).
fn exact_scores(spec: &TwinSpec, x: &[f64]) -> Vec<BigRational> {
    let q = |v: f64| BigRational::from_float(v).expect("finite operand");
    let zero = q(0.0);
    let mut t: Vec<Vec<BigRational>> = Vec::with_capacity(spec.ops.len() + 1);
    t.push(x.iter().map(|&v| q(v)).collect());
    for op in &spec.ops {
        let out: Vec<BigRational> = match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let src = &t[*input];
                let (co, ci, kh, kw) = *kernel;
                let (_, ih, iw) = *ishape;
                let (_, oh, ow) = *oshape;
                let mut out = Vec::with_capacity(co * oh * ow);
                for oc in 0..co {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = q(bias[oc]);
                            for c in 0..ci {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oy * stride.0 + ky) as isize - pads.0 as isize;
                                        let ix = (ox * stride.1 + kx) as isize - pads.1 as isize;
                                        if iy >= 0
                                            && (iy as usize) < ih
                                            && ix >= 0
                                            && (ix as usize) < iw
                                        {
                                            let w = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            if w != 0.0 {
                                                let s = &src[c * ih * iw
                                                    + (iy as usize) * iw
                                                    + ix as usize];
                                                acc += q(w) * s;
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
            TwinOpSpec::ConvTranspose {
                input,
                weight,
                bias,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let src = &t[*input];
                let (co, ci, kh, kw) = *kernel;
                let (_, ih, iw) = *ishape;
                let (_, oh, ow) = *oshape;
                let mut out: Vec<BigRational> =
                    (0..co * oh * ow).map(|i| q(bias[i / (oh * ow)])).collect();
                for c in 0..ci {
                    for iy in 0..ih {
                        for ix in 0..iw {
                            let xv = &src[c * ih * iw + iy * iw + ix];
                            for oc in 0..co {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let ny = iy * stride.0 + ky;
                                        let nx = ix * stride.1 + kx;
                                        if ny < pads.0 || nx < pads.1 {
                                            continue;
                                        }
                                        let (oy, ox) = (ny - pads.0, nx - pads.1);
                                        if oy < oh && ox < ow {
                                            let w = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            out[(oc * oh + oy) * ow + ox] += q(w) * xv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                out
            }
            TwinOpSpec::ChannelAffine {
                input,
                scale,
                shift,
                shape,
                ..
            } => {
                let hw = shape.1 * shape.2;
                t[*input]
                    .iter()
                    .enumerate()
                    .map(|(i, v)| q(shift[i / hw]) + q(scale[i / hw]) * v)
                    .collect()
            }
            TwinOpSpec::Relu { input } => t[*input]
                .iter()
                .map(|v| if *v > zero { v.clone() } else { zero.clone() })
                .collect(),
            TwinOpSpec::Add { lhs, rhs } => {
                t[*lhs].iter().zip(&t[*rhs]).map(|(a, b)| a + b).collect()
            }
            TwinOpSpec::Flatten { input } => t[*input].clone(),
            TwinOpSpec::Gemm {
                input,
                weight,
                bias,
                shape,
                ..
            } => {
                let src = &t[*input];
                let (no, ni) = *shape;
                (0..no)
                    .map(|o| {
                        let mut acc = q(bias[o]);
                        for (i, s) in src.iter().enumerate().take(ni) {
                            let w = weight[o * ni + i];
                            if w != 0.0 {
                                acc += q(w) * s;
                            }
                        }
                        acc
                    })
                    .collect()
            }
        };
        t.push(out);
    }
    t.pop().expect("final tensor")
}

/// THE EXACT-RATIONAL FALSIFIER. For every fixture, box shape (a POINT box —
/// the relaxation is exact there and only certified rounding separates the
/// bound from the truth — and three growing radii) and envelope lane, the
/// published root bound `root_eval.dj[k]` must satisfy
/// `dj[k] <= exact_margin_k(x)` IN EXACT ARITHMETIC at the center, both box
/// corners, random corners and random interior points. The per-column lane is
/// run as the control and must hold too; the smallest exact slack is printed
/// so a reader can see the point boxes are genuinely tight.
#[test]
fn eel_exact_rational_falsifier_encloses_sampled_forwards() {
    let mut rng = StdRng::seed_from_u64(0xFA15_1F1E);
    let mut checks = 0usize;
    let mut min_slack_point = f64::INFINITY;
    for (name, spec, lo0, hi0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let n_in = spec.n_in;
        for trial in 0..3 {
            let center: Vec<f64> = (0..n_in)
                .map(|i| {
                    let u: f64 = rng.random_range(0.0..1.0);
                    lo0[i] + u * (hi0[i] - lo0[i])
                })
                .collect();
            for rad in [0.0, 1e-6, 1e-3, 0.05] {
                let lo: Vec<f64> = center.iter().map(|c| c - rad).collect();
                let hi: Vec<f64> = center.iter().map(|c| c + rad).collect();
                let root =
                    RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
                let (t, _, adv) = tiny_targets(&net, &lo, &hi);
                let mut pts: Vec<Vec<f64>> = vec![center.clone(), lo.clone(), hi.clone()];
                if rad > 0.0 {
                    for _ in 0..6 {
                        pts.push(
                            (0..n_in)
                                .map(|i| {
                                    if rng.random_range(0.0..1.0) < 0.5 {
                                        lo[i]
                                    } else {
                                        hi[i]
                                    }
                                })
                                .collect(),
                        );
                        pts.push(
                            (0..n_in)
                                .map(|i| lo[i] + rng.random_range(0.0..1.0) * (hi[i] - lo[i]))
                                .collect(),
                        );
                    }
                }
                let exact: Vec<Vec<BigRational>> = pts
                    .iter()
                    .map(|x| {
                        let s = exact_scores(&spec, x);
                        adv.iter().map(|&j| &s[t] - &s[j]).collect()
                    })
                    .collect();
                for lane in [ErrLane::PerCol, ErrLane::Envelope, ErrLane::PerChannel] {
                    let eng = BackwardEngine::new(&net, &root)
                        .with_err_lane(lane)
                        .with_err_min_cols(1);
                    let re = super::super::bab::root_eval(&eng, &net, t, &adv).expect("root_eval");
                    for (p, m) in exact.iter().enumerate() {
                        for (k, mk) in m.iter().enumerate() {
                            let dj = BigRational::from_float(re.dj[k]).expect("finite bound");
                            assert!(
                                &dj <= mk,
                                "FALSIFIED: {name} trial {trial} rad {rad:e} lane {lane:?}: \
                                 published dj[{k}] = {:e} exceeds the EXACT margin at point {p}",
                                re.dj[k]
                            );
                            checks += 1;
                            if rad == 0.0 && lane != ErrLane::PerCol {
                                use num_traits::ToPrimitive;
                                let slack = (mk - &dj).to_f64().unwrap_or(f64::INFINITY);
                                min_slack_point = min_slack_point.min(slack);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checks > 500, "falsifier ran only {checks} exact checks");
    eprintln!(
        "[mr-errlane-test] exact-rational falsifier: {checks} exact checks, 0 violations; \
         smallest exact slack on a POINT box under an envelope = {min_slack_point:e}"
    );
}

/// The discriminating running-bias oracle (`oracle_deep_conv_bias_running_rounding`,
/// which caught the real `eb` under-count) with the envelope lanes carrying
/// `eb` and `E`: zero violations of `dj <= true margin` against the
/// double-double forward.
#[test]
fn eel_adversarial_bias_rounding_oracle_holds_under_envelopes() {
    let mut cfg = AdvCfg {
        cin: 3,
        hs: 4,
        ws: 4,
        cw: 256,
        k_tiny: 12,
        tiny_bias: 0.0,
        relu_bias: 5.0,
        gemm1_bias: 3.0e6,
        n_y: 24,
        n_out: 10,
        conv_wscale: 1.0 / 256.0,
        gemm1_wscale: 1.0e-4,
        gemm2_wscale: 0.5,
        t_bias: 0.0,
        seed: 0xE3_1A5E,
    };
    let t = 0usize;
    let adv: Vec<usize> = (1..cfg.n_out).collect();
    let (l, sigma) = measure_l_and_sigma_lr(&cfg, t, &adv);
    let width = cfg.cw * cfg.hs * cfg.ws;
    let n_tiny = (cfg.k_tiny * width) as f64;
    let dom_sigma = sigma.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let lr_avg = dom_sigma / n_tiny;
    let ulp_half = 0.5 * (super::super::rounding::next_up(l) - l);
    cfg.tiny_bias = -(0.25 * ulp_half) / lr_avg;
    cfg.t_bias = -l;
    let spec = adversarial_spec(&cfg);
    let net = TwinNet::compile(&spec).expect("compile");
    let n_in = spec.n_in;
    let lo = vec![-1.0e-9; n_in];
    let hi = vec![1.0e-9; n_in];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let mut rng = StdRng::seed_from_u64(0x0BAD_E1E1);
    let mut pts: Vec<Vec<f64>> = vec![vec![0.0; n_in], lo.clone(), hi.clone()];
    for _ in 0..8 {
        pts.push(
            (0..n_in)
                .map(|i| {
                    if rng.random_range(0.0..1.0) < 0.5 {
                        lo[i]
                    } else {
                        hi[i]
                    }
                })
                .collect(),
        );
    }
    let truth: Vec<Vec<f64>> = pts.iter().map(|x| dd_margins(&spec, x, t, &adv)).collect();
    for lane in [ErrLane::PerCol, ErrLane::Envelope, ErrLane::PerChannel] {
        let eng = BackwardEngine::new(&net, &root)
            .with_err_lane(lane)
            .with_err_min_cols(1);
        let re = super::super::bab::root_eval(&eng, &net, t, &adv).expect("root_eval");
        let mut violations = 0usize;
        let mut worst = f64::NEG_INFINITY;
        for m in &truth {
            for (k, &mk) in m.iter().enumerate() {
                worst = worst.max(re.dj[k] - mk);
                if re.dj[k] > mk + 1e-12 {
                    violations += 1;
                }
            }
        }
        eprintln!(
            "[mr-errlane-test] adversarial oracle lane={lane:?}: violations={violations} \
             worst(dj - true)={worst:e}"
        );
        assert_eq!(
            violations, 0,
            "ENCLOSURE ORACLE under {lane:?}: certified dj exceeds the true margin"
        );
    }
}

/// `off` IS TODAY'S LANE. Byte-identity is asserted on everything `off` could
/// be confused with: (1) the declaration and every policy default resolve to
/// `off`; (2) the publish table never engages below the floor or in `audit`;
/// (3) per pass, an `off` engine, an `envelope` engine below its width floor
/// and an `audit` engine publish BYTE-IDENTICAL `PassOut`s over margin /
/// identity / dense / armed-domain / prefix / stacked passes; (4) per lane,
/// `MarginRowBab::run` under `audit` returns the `off` verdict bit-for-bit
/// (root bound bits, expansions, stop reason) on a closing and a falsified
/// instance, and the envelope lanes keep the moat.
#[test]
fn eel_off_is_bit_identical() {
    // (1) Defaults.
    assert_eq!(ErrLane::default(), ErrLane::PerCol);
    assert_eq!(
        super::super::RootPrecisionRuntimePolicy::default().err_lane,
        ErrLane::PerCol
    );
    let frozen = ny_levers::LeverSet::freeze_with(ny_levers::all(), |_| None);
    assert_eq!(
        super::super::MarginRowRuntimePolicy::from_levers(&frozen)
            .expect("complete registry")
            .root_precision
            .err_lane,
        ErrLane::PerCol,
        "an empty frozen LeverSet must resolve the declaration default `off`"
    );
    assert_eq!(
        ny_levers::decls::margin_row::MARGIN_ROW_ERR_ENVELOPE.default,
        ny_levers::DefaultSpec::Text("off")
    );
    // (2) Publish table.
    for r in [0usize, 1, 3, ENV_MIN_COLS - 1, ENV_MIN_COLS, 192] {
        assert_eq!(ErrLane::PerCol.publish(r, ENV_MIN_COLS), PassLane::PerCol);
        assert_eq!(ErrLane::Audit.publish(r, ENV_MIN_COLS), PassLane::PerCol);
        assert_eq!(
            ErrLane::Envelope.publish(r, ENV_MIN_COLS),
            if r >= ENV_MIN_COLS {
                PassLane::Envelope { per_channel: false }
            } else {
                PassLane::PerCol
            }
        );
        assert_eq!(
            ErrLane::PerChannel.publish(r, ENV_MIN_COLS),
            if r >= 1 {
                PassLane::Envelope { per_channel: true }
            } else {
                PassLane::PerCol
            }
        );
    }
    // (3) Per pass.
    let same = |a: &PassOut, b: &PassOut, what: &str| {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(a.a.raw_dim(), b.a.raw_dim(), "{what}");
        assert_eq!(
            bits(a.a.as_slice().expect("layout")),
            bits(b.a.as_slice().expect("layout")),
            "{what}: a"
        );
        assert_eq!(
            a.e.as_ref().map(|e| bits(e.as_slice().expect("layout"))),
            b.e.as_ref().map(|e| bits(e.as_slice().expect("layout"))),
            "{what}: e"
        );
        assert_eq!(bits(&a.b), bits(&b.b), "{what}: b");
        assert_eq!(bits(&a.eb), bits(&b.eb), "{what}: eb");
        assert_eq!(
            a.coll
                .as_ref()
                .map(|m| m.values().map(|v| bits(v)).collect::<Vec<_>>()),
            b.coll
                .as_ref()
                .map(|m| m.values().map(|v| bits(v)).collect::<Vec<_>>()),
            "{what}: coll"
        );
    };
    let mut passes = 0usize;
    for (name, spec, lo, hi) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let off = BackwardEngine::new(&net, &root);
        assert_eq!(
            off.err_lane(),
            ErrLane::PerCol,
            "{name}: new() must be dark"
        );
        let floor = BackwardEngine::new(&net, &root).with_err_lane(ErrLane::Envelope);
        let audit = BackwardEngine::new(&net, &root).with_err_lane(ErrLane::Audit);
        let (t, _, adv) = tiny_targets(&net, &lo, &hi);
        let mb = MarginBatch::new(&net, t, &adv).expect("mb");
        let (al, au) = off.y_rows(None).expect("y rows");
        let ybox = YBox::from_rows(&off, &al, &au);
        let gates = head_gates(&ybox, RoundMode::Outward);
        let margin = margin_seed(&mb, &gates, &ybox, RoundMode::Outward).seed;
        let seeds = vec![
            ("margin", margin),
            ("identity", off.identity_seed()),
            ("dense7", dense_seed(net.n_y, 7, 5)),
        ];
        for (sname, seed) in &seeds {
            assert!(
                seed.s.ncols() < ENV_MIN_COLS,
                "fixture must sit below the floor"
            );
            let (dom, exc) = armed_domain(&root, seed.s.ncols());
            for armed in [false, true] {
                for dir in [LaneDir::Lower, LaneDir::Upper] {
                    let (d, x) = if armed {
                        (Some(&dom), Some(&exc))
                    } else {
                        (None, None)
                    };
                    let collect = Collect {
                        unst_abs: true,
                        rows: None,
                        unst_rows: false,
                    };
                    let what = format!("{name}/{sname}/armed={armed}/{dir:?}");
                    let a = off.run_collect(seed, d, dir, x, collect).expect("off");
                    let b = floor.run_collect(seed, d, dir, x, collect).expect("floor");
                    let c = audit.run_collect(seed, d, dir, x, collect).expect("audit");
                    same(&a, &b, &format!("{what} envelope-below-floor"));
                    same(&a, &c, &format!("{what} audit"));
                    passes += 1;
                }
            }
        }
        for rec in &root.layers {
            let n_pre = rec.n;
            let r = n_pre.min(9);
            let mut s = Array2::<f64>::zeros((n_pre, r));
            for c in 0..r {
                s[[(c * 5) % n_pre, c]] = 1.0;
            }
            let seed = Seed { s, e: None };
            let a = off.run_prefix(&seed, rec.op, LaneDir::Lower).expect("off");
            let b = floor
                .run_prefix(&seed, rec.op, LaneDir::Lower)
                .expect("floor");
            let c = audit
                .run_prefix(&seed, rec.op, LaneDir::Lower)
                .expect("audit");
            same(&a, &b, &format!("{name}/prefix envelope-below-floor"));
            same(&a, &c, &format!("{name}/prefix audit"));
            passes += 1;
        }
    }
    assert!(passes >= 40, "only {passes} passes compared");

    // (4) Per lane: `audit` reproduces the `off` tree bit-for-bit; every
    // envelope lane keeps the moat and still closes the closing instance.
    // The `lane_verdicts_unsat_and_fail_closed` fixture (known to verify).
    let mut rng = StdRng::seed_from_u64(29);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let (t, worst, adv) = tiny_targets(&net, &lo, &hi);
    let build = |lane: ErrLane| {
        let mut root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        root.root_precision_policy.err_lane = lane;
        root
    };
    let cfg = || BabConfig {
        lru_cap: 64,
        frontier: 1,
        max_expansions: 200,
        ..BabConfig::default()
    };
    let root_off = build(ErrLane::PerCol);
    let off_close = MarginRowBab::run(&net, &root_off, t, &adv, cfg());
    let off_moat = MarginRowBab::run(&net, &root_off, worst, &[t], cfg());
    let root_audit = build(ErrLane::Audit);
    let audit_close = MarginRowBab::run(&net, &root_audit, t, &adv, cfg());
    let audit_moat = MarginRowBab::run(&net, &root_audit, worst, &[t], cfg());
    assert_eq!(
        verdict_bits(&off_close),
        verdict_bits(&audit_close),
        "audit must publish the off tree bit-for-bit (closing instance)"
    );
    assert_eq!(
        verdict_bits(&off_moat),
        verdict_bits(&audit_moat),
        "audit must publish the off tree bit-for-bit (falsified instance)"
    );
    assert!(
        matches!(off_close, MarginRowOutcome::Unsat(..)),
        "fixture must close under off"
    );
    for lane in ENVELOPES {
        let root = build(lane);
        assert!(
            matches!(
                MarginRowBab::run(&net, &root, worst, &[t], cfg()),
                MarginRowOutcome::Unknown { .. }
            ),
            "MOAT VIOLATION: {lane:?} verified a falsified instance"
        );
        assert!(
            matches!(
                MarginRowBab::run(&net, &root, t, &adv, cfg()),
                MarginRowOutcome::Unsat(..)
            ),
            "{lane:?} lost the closing instance: the envelope is too loose to be armed"
        );
    }
}

/// Envelope bookkeeping: regrouping only ever dominates, channel groups are
/// the conv spatial sizes and partition every tensor.
#[test]
fn eel_regroup_and_channel_groups_are_dominating_partitions() {
    let e = Array2::from_shape_fn((12, 3), |(j, c)| ((j * 3 + c) % 7) as f64 * 1e-15);
    let row = errlane::EnvErr::from_matrix(&e, 1);
    for j in 0..12 {
        for c in 0..3 {
            assert!(row.at(j) >= e[[j, c]]);
        }
    }
    let coarse = row.clone().regroup(4);
    assert_eq!(coarse.grp, 4);
    assert_eq!(coarse.v.len(), 3);
    for j in 0..12 {
        assert!(coarse.at(j) >= row.at(j), "coarsening must dominate");
    }
    let fine = coarse.clone().regroup(2);
    for j in 0..12 {
        assert!(
            fine.at(j) >= row.at(j),
            "refining a coarse envelope must dominate"
        );
        assert_eq!(fine.at(j).to_bits(), coarse.at(j).to_bits());
    }
    // A non-partitioning request falls back to per-row (always sound).
    assert_eq!(row.regroup(5).grp, 1);
    let b = coarse.broadcast(3);
    for j in 0..12 {
        for c in 0..3 {
            assert_eq!(b[[j, c]].to_bits(), coarse.at(j).to_bits());
        }
    }
    let mut rng = StdRng::seed_from_u64(7);
    let net = TwinNet::compile(&tiny_spec(&mut rng, 0.5)).expect("compile");
    let g = errlane::channel_groups(&net);
    // tiny_spec: t0 (2,6,6) -> conv1 (4,3,3) -> relu -> conv2 -> add -> relu
    // -> flatten -> head.
    assert_eq!(g[0], 36);
    for tid in 1..=5 {
        assert_eq!(g[tid], 9, "tensor {tid}");
    }
    assert_eq!(g[6], 1, "the flatten output groups per row");
    for (tid, &gs) in g.iter().enumerate() {
        assert_eq!(net.tsize[tid] % gs, 0, "tensor {tid} is not partitioned");
    }
}

/// The stop receipt keeps a fixed schema (a grep distinguishes zero from a
/// binary that lacks the lane) and the armed line names its mode.
#[test]
fn eel_observables_have_fixed_schema() {
    let line = errlane::stop_line();
    assert!(line.starts_with("[mr-errlane] done mode="), "{line}");
    for key in [
        " passes=",
        " eligible=",
        " env_passes=",
        " pc_passes=",
        " audited=",
        " max_pen_env=",
        " max_pen_col=",
        " max_pen_pc=",
        " aud_row_passes=",
        " aud_row_max_dpen=",
        " aud_row_max_loosen=",
        " aud_row_max_rel=",
        " aud_row_min_pos=",
        " aud_row_pos_lost=",
        " aud_row_dom_viol=",
        " aud_row_lane_mismatch=",
        " aud_pc_passes=",
        " aud_pc_max_dpen=",
        " aud_pc_pos_lost=",
        " aud_pc_dom_viol=",
    ] {
        assert!(line.contains(key), "missing {key}: {line}");
    }
    assert!(!line.contains('\n'));
    for lane in [ErrLane::Envelope, ErrLane::PerChannel, ErrLane::Audit] {
        let armed = errlane::announce(lane);
        assert_eq!(
            armed,
            format!(
                "[mr-errlane] armed mode={} min_cols={ENV_MIN_COLS} source=policy",
                lane.token()
            )
        );
    }
    assert_eq!(
        errlane::announce(ErrLane::PerCol),
        "[mr-errlane] dark mode=off"
    );
}
