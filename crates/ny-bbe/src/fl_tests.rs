// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M8 forward-linear search tests: the f64 reference instantiation CONTAINS
//! sampled truth on random nets exercising every supported op; f32 tracks it;
//! the input multipliers are the affine function's gradient where the
//! relaxation is exact; a domain's outputs depend on neither its batch nor
//! the thread count; malformed, unsupported, foreign, late and oversized
//! requests are refused typed; shared weights upload once; the working-set
//! estimate covers the real allocation.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::search::{
    DeviceLease, FlBatch, FlFamily, FlHandle, FlLinear, FlOp, FlOpConsts, FlOut, FlProblem,
    FlProgram, Refusal, SearchBackend,
};

use crate::fl::{self, FlUpload, FL_LANES};
use crate::CpuF32Backend;

/// Deterministic LCG in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn sym(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

fn arc(v: Vec<f64>) -> Arc<[f64]> {
    v.into()
}

fn lin(src: usize, n_in: usize, n_out: usize, w: Arc<[f64]>) -> FlOp {
    FlOp::Linear(FlLinear {
        src: src as u32,
        n_in: n_in as u32,
        n_out: n_out as u32,
        w,
    })
}

fn weights(rng: &mut Lcg, n_out: usize, n_in: usize, scale: f64) -> Arc<[f64]> {
    arc((0..n_out * n_in).map(|_| rng.sym() * scale).collect())
}

fn bias(rng: &mut Lcg, n: usize, shift: f64, width: f64) -> FlOpConsts {
    let lo: Vec<f64> = (0..n).map(|_| rng.sym() * 0.3 + shift).collect();
    let hi: Vec<f64> = lo.iter().map(|&l| l + width * rng.unit()).collect();
    FlOpConsts::Bias {
        lo: arc(lo),
        hi: arc(hi),
    }
}

/// A random net over every op the CPU backend supports:
///
/// ```text
///  0 Input d            1 Linear 0->6        2 Relu 1
///  3 Linear 2->5        4 Sigmoid 3          5 Slice 2 [1..5)
///  6 Linear 4->4        7 Add 5 6            8 Sub 7 5
///  9 AddConst 8        10 RevSubConst 9     11 MulConst 10
/// 12 Identity 11       13 Concat [12, 0]    14 Sparse (3 out; 13 and 2)
/// 15 Relu 14           16 Linear 15->2 (seed)
/// ```
///
/// `clauses` clauses differ in every constant (biases with interval width
/// `width`, `*Const` vectors, sparse coefficients and constants).
fn random_family(rng: &mut Lcg, d: usize, clauses: usize, width: f64) -> FlFamily {
    let sizes = vec![d, 6, 6, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4 + d, 3, 3, 2];
    let terms: Vec<Vec<(u32, u32)>> = vec![
        vec![(13, 0), (13, 3), (2, 5)],
        vec![(13, 4), (2, 0)],
        vec![(13, 1), (13, 2), (13, (4 + d - 1) as u32), (2, 2)],
    ];
    let n_terms: usize = terms.iter().map(Vec::len).sum();
    let program = FlProgram {
        d,
        sizes,
        seed: 16,
        ops: vec![
            FlOp::Input,
            lin(0, d, 6, weights(rng, 6, d, 1.2)),
            FlOp::Relu { src: 1 },
            lin(2, 6, 5, weights(rng, 5, 6, 0.8)),
            FlOp::Sigmoid { src: 3 },
            FlOp::Slice { src: 2, start: 1 },
            lin(4, 5, 4, weights(rng, 4, 5, 1.0)),
            FlOp::Add { a: 5, b: 6 },
            FlOp::Sub { a: 7, b: 5 },
            FlOp::AddConst { src: 8 },
            FlOp::RevSubConst { src: 9 },
            FlOp::MulConst { src: 10 },
            FlOp::Identity { src: 11 },
            FlOp::Concat { srcs: vec![12, 0] },
            FlOp::Sparse { terms },
            FlOp::Relu { src: 14 },
            lin(15, 3, 2, weights(rng, 2, 3, 1.0)),
        ],
    };
    let clauses = (0..clauses)
        .map(|_| {
            let k = |rng: &mut Lcg| FlOpConsts::K(arc((0..4).map(|_| rng.sym()).collect()));
            let k_lo: Vec<f64> = (0..3).map(|_| rng.sym() * 0.2).collect();
            let k_hi: Vec<f64> = k_lo.iter().map(|&l| l + width * rng.unit()).collect();
            vec![
                FlOpConsts::None,
                bias(rng, 6, 0.0, width),
                FlOpConsts::None,
                bias(rng, 5, 0.0, width),
                FlOpConsts::None,
                FlOpConsts::None,
                bias(rng, 4, 0.0, width),
                FlOpConsts::None,
                FlOpConsts::None,
                k(rng),
                k(rng),
                k(rng),
                FlOpConsts::None,
                FlOpConsts::None,
                FlOpConsts::Sparse {
                    coef: arc((0..n_terms).map(|_| rng.sym() * 1.5).collect()),
                    k_lo: arc(k_lo),
                    k_hi: arc(k_hi),
                },
                FlOpConsts::None,
                bias(rng, 2, 0.0, width),
            ]
        })
        .collect();
    FlFamily { program, clauses }
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Exact-ish f64 point evaluation of every node; each interval constant is
/// taken at `t` of its way from `lo` to `hi` (the bound must hold for every
/// member of the constant set).
fn eval(fam: &FlFamily, clause: usize, x: &[f64], t: f64) -> Vec<Vec<f64>> {
    let p = &fam.program;
    let consts = &fam.clauses[clause];
    let pick = |lo: f64, hi: f64| lo + (hi - lo) * t;
    let mut vals: Vec<Vec<f64>> = Vec::with_capacity(p.ops.len());
    for (k, op) in p.ops.iter().enumerate() {
        let n = p.sizes[k];
        let v: Vec<f64> = match (op, &consts[k]) {
            (FlOp::Input, _) => x.to_vec(),
            (FlOp::Linear(l), FlOpConsts::Bias { lo, hi }) => {
                let s = &vals[l.src as usize];
                let ni = l.n_in as usize;
                (0..n)
                    .map(|i| {
                        pick(lo[i], hi[i]) + (0..ni).map(|j| l.w[i * ni + j] * s[j]).sum::<f64>()
                    })
                    .collect()
            }
            (FlOp::Sparse { terms }, FlOpConsts::Sparse { coef, k_lo, k_hi }) => {
                let mut at = 0usize;
                (0..n)
                    .map(|j| {
                        let mut acc = pick(k_lo[j], k_hi[j]);
                        for &(s, i) in &terms[j] {
                            acc += coef[at] * vals[s as usize][i as usize];
                            at += 1;
                        }
                        acc
                    })
                    .collect()
            }
            (FlOp::Relu { src }, _) => vals[*src as usize].iter().map(|v| v.max(0.0)).collect(),
            (FlOp::Sigmoid { src }, _) => vals[*src as usize].iter().map(|&v| sig(v)).collect(),
            (FlOp::Slice { src, start }, _) => {
                vals[*src as usize][*start as usize..*start as usize + n].to_vec()
            }
            (FlOp::Concat { srcs }, _) => srcs
                .iter()
                .flat_map(|&s| vals[s as usize].clone())
                .collect(),
            (FlOp::Identity { src }, _) => vals[*src as usize].clone(),
            (FlOp::Add { a, b }, _) => vals[*a as usize]
                .iter()
                .zip(&vals[*b as usize])
                .map(|(p, q)| p + q)
                .collect(),
            (FlOp::Sub { a, b }, _) => vals[*a as usize]
                .iter()
                .zip(&vals[*b as usize])
                .map(|(p, q)| p - q)
                .collect(),
            (FlOp::AddConst { src }, FlOpConsts::K(kv)) => vals[*src as usize]
                .iter()
                .zip(kv.iter())
                .map(|(p, q)| p + q)
                .collect(),
            (FlOp::RevSubConst { src }, FlOpConsts::K(kv)) => vals[*src as usize]
                .iter()
                .zip(kv.iter())
                .map(|(p, q)| q - p)
                .collect(),
            (FlOp::MulConst { src }, FlOpConsts::K(kv)) => vals[*src as usize]
                .iter()
                .zip(kv.iter())
                .map(|(p, q)| p * q)
                .collect(),
            _ => panic!("eval: op {k} unsupported by the test evaluator"),
        };
        vals.push(v);
    }
    vals
}

fn objective_value(fam: &FlFamily, clause: usize, x: &[f64], t: f64, obj: &[f64]) -> f64 {
    let vals = eval(fam, clause, x, t);
    vals[fam.program.seed]
        .iter()
        .zip(obj)
        .map(|(v, o)| v * o)
        .sum()
}

/// A random sub-box of `[-1, 1]^d` with some narrow and some point coordinates.
fn random_box(rng: &mut Lcg, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lo = Vec::with_capacity(d);
    let mut hi = Vec::with_capacity(d);
    for _ in 0..d {
        let c = rng.sym() * 0.8;
        let r = match rng.unit() {
            u if u < 0.1 => 0.0,
            u if u < 0.3 => 1e-3 * rng.unit(),
            _ => 0.4 * rng.unit(),
        };
        lo.push(c - r);
        hi.push(c + r);
    }
    (lo, hi)
}

struct Run {
    lb: Vec<f32>,
    scale: Vec<f32>,
    grad: Vec<f32>,
}

/// Bound `count` domains of family `f` (every domain asks for `rows[b]`).
#[allow(clippy::too_many_arguments)]
fn run_real<R: crate::real::Real>(
    problem: &FlProblem,
    f: usize,
    lo: &[f64],
    hi: &[f64],
    clause: &[u32],
    rows: &[Vec<u32>],
    threads: usize,
) -> Result<Run, Refusal> {
    let lease = DeviceLease { bytes: 1 << 34 };
    let h = fl::upload::<R>("test", problem, &lease, threads)?;
    let d = problem.families[f].program.d;
    let count = lo.len() / d;
    let mut off = vec![0u32];
    let mut flat = Vec::new();
    for r in rows {
        flat.extend_from_slice(r);
        off.push(flat.len() as u32);
    }
    let batch = FlBatch {
        family: f as u32,
        count,
        d,
        lo,
        hi,
        clause,
        obj_offsets: &off,
        obj_rows: &flat,
        deadline: None,
    };
    let mut out = FlOut::default();
    fl::forward_linear_batch::<R>(&h, &batch, &mut out, threads)?;
    let view = out.hint().into_ingest();
    assert!(view.well_formed());
    let n = view.rows();
    Ok(Run {
        lb: (0..n).map(|r| view.lb(r)).collect(),
        scale: (0..n).map(|r| view.scale(r)).collect(),
        grad: (0..n).flat_map(|r| view.grad(r).to_vec()).collect(),
    })
}

fn problem_of(fams: Vec<FlFamily>, rng: &mut Lcg, rows: usize) -> FlProblem {
    let width = fams[0].program.sizes[fams[0].program.seed];
    let objectives = (0..rows)
        .map(|_| (0..width).map(|_| rng.sym()).collect())
        .collect();
    FlProblem {
        families: fams,
        objectives,
    }
}

/// Every sampled point (and every box corner) of every domain is at or above
/// the f64 reference's lower bound for every clause-constant member, on random
/// nets exercising every supported op.
#[test]
fn fl_f64_reference_bounds_contain_sampled_truth() {
    let mut rng = Lcg(0x5eed_f00d);
    let mut checked = 0usize;
    for net in 0..6 {
        let d = 2 + net % 3;
        let fam = random_family(&mut rng, d, 2, if net % 2 == 0 { 0.0 } else { 0.05 });
        let problem = problem_of(vec![fam], &mut rng, 3);
        problem.validate().expect("valid problem");
        let boxes: Vec<(Vec<f64>, Vec<f64>)> = (0..3).map(|_| random_box(&mut rng, d)).collect();
        let lo: Vec<f64> = boxes.iter().flat_map(|b| b.0.clone()).collect();
        let hi: Vec<f64> = boxes.iter().flat_map(|b| b.1.clone()).collect();
        let clause: Vec<u32> = (0..3).map(|b| (b % 2) as u32).collect();
        let rows: Vec<Vec<u32>> = vec![vec![0, 1, 2]; 3];
        let run = run_real::<f64>(&problem, 0, &lo, &hi, &clause, &rows, 2).expect("runs");
        let fam = &problem.families[0];
        for (b, (bl, bh)) in boxes.iter().enumerate() {
            for s in 0..3 {
                let r = b * 3 + s;
                let lb = f64::from(run.lb[r]);
                let scale = f64::from(run.scale[r]);
                assert!(lb.is_finite(), "net {net} domain {b} row {s}: lb {lb}");
                let obj = &problem.objectives[s];
                let tol = 1e-6 * (1.0 + scale);
                let mut vmin = f64::INFINITY;
                for i in 0..2000 {
                    let x: Vec<f64> = (0..d)
                        .map(|k| {
                            if i < 1 << d {
                                if (i >> k) & 1 == 1 {
                                    bh[k]
                                } else {
                                    bl[k]
                                }
                            } else {
                                bl[k] + (bh[k] - bl[k]) * rng.unit()
                            }
                        })
                        .collect();
                    let t = [0.0, 1.0, rng.unit()][i % 3];
                    let v = objective_value(fam, clause[b] as usize, &x, t, obj);
                    vmin = vmin.min(v);
                    checked += 1;
                }
                assert!(
                    lb <= vmin + tol,
                    "CONTAINMENT: net {net} domain {b} row {s}: lb {lb} > sampled min {vmin}"
                );
            }
        }
    }
    assert!(checked >= 6 * 3 * 3 * 2000, "VOID: {checked} samples");
}

/// The shipped f32 search tracks its own f64 reference on the same domains.
#[test]
fn fl_f32_tracks_the_f64_reference() {
    let mut rng = Lcg(77);
    let mut compared = 0usize;
    for net in 0..5 {
        let d = 2 + net % 3;
        let fam = random_family(&mut rng, d, 3, 0.02);
        let problem = problem_of(vec![fam], &mut rng, 2);
        let count = 21;
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for _ in 0..count {
            let (l, h) = random_box(&mut rng, d);
            lo.extend(l);
            hi.extend(h);
        }
        let clause: Vec<u32> = (0..count).map(|b| (b % 3) as u32).collect();
        let rows: Vec<Vec<u32>> = (0..count)
            .map(|b| if b % 2 == 0 { vec![0, 1] } else { vec![1] })
            .collect();
        let r64 = run_real::<f64>(&problem, 0, &lo, &hi, &clause, &rows, 1).expect("f64");
        let r32 = run_real::<f32>(&problem, 0, &lo, &hi, &clause, &rows, 1).expect("f32");
        for r in 0..r64.lb.len() {
            let (a, b) = (f64::from(r32.lb[r]), f64::from(r64.lb[r]));
            let s = f64::from(r64.scale[r]).max(1.0);
            assert!(
                (a - b).abs() <= 1e-4 * s,
                "net {net} row {r}: f32 {a} vs f64 {b} (scale {s})"
            );
            compared += 1;
        }
    }
    assert!(compared > 50, "VOID: {compared} rows compared");
}

/// Where every ReLU is stable on the box the relaxation is exact, the
/// objective is affine there, and the returned input multipliers are its
/// gradient (central differences at the box centre).
#[test]
fn fl_gradients_match_central_differences() {
    let mut rng = Lcg(4242);
    let d = 3;
    // 1: Linear 0->5 with biases far above 0 (always active);
    // 3: Linear 2->4 with biases far below 0 (always inactive) mixed with
    //    the active branch through Add, Sub, AddConst, MulConst, Sparse.
    let program = FlProgram {
        d,
        sizes: vec![d, 5, 5, 4, 4, 4, 4, 4, 4 + d, 2, 2],
        seed: 10,
        ops: vec![
            FlOp::Input,
            lin(0, d, 5, weights(&mut rng, 5, d, 0.5)),
            FlOp::Relu { src: 1 },
            lin(2, 5, 4, weights(&mut rng, 4, 5, 0.5)),
            FlOp::Relu { src: 3 },
            FlOp::Slice { src: 2, start: 1 },
            FlOp::Sub { a: 5, b: 4 },
            FlOp::MulConst { src: 6 },
            FlOp::Concat { srcs: vec![7, 0] },
            FlOp::Sparse {
                terms: vec![vec![(8, 0), (8, 4), (8, 6)], vec![(8, 2), (8, 5)]],
            },
            FlOp::AddConst { src: 9 },
        ],
    };
    let consts = vec![
        FlOpConsts::None,
        bias(&mut rng, 5, 20.0, 0.0),
        FlOpConsts::None,
        bias(&mut rng, 4, -200.0, 0.0),
        FlOpConsts::None,
        FlOpConsts::None,
        FlOpConsts::None,
        FlOpConsts::K(arc(vec![1.5, -0.5, 2.0, -1.0])),
        FlOpConsts::None,
        FlOpConsts::Sparse {
            coef: arc(vec![1.0, -2.0, 0.5, 3.0, -1.0]),
            k_lo: arc(vec![0.1, -0.2]),
            k_hi: arc(vec![0.1, -0.2]),
        },
        FlOpConsts::K(arc(vec![0.25, 0.75])),
    ];
    let fam = FlFamily {
        program,
        clauses: vec![consts],
    };
    let problem = FlProblem {
        families: vec![fam],
        objectives: vec![vec![1.0, -0.5], vec![-2.0, 1.0]],
    };
    let lo = vec![-0.3, 0.1, -0.2];
    let hi = vec![0.2, 0.4, 0.3];
    let run = run_real::<f64>(&problem, 0, &lo, &hi, &[], &[vec![0, 1]], 1).expect("runs");
    let centre: Vec<f64> = lo.iter().zip(&hi).map(|(a, b)| 0.5 * (a + b)).collect();
    let fam = &problem.families[0];
    for (s, obj) in problem.objectives.iter().enumerate() {
        for k in 0..d {
            let h = 1e-4;
            let (mut xp, mut xm) = (centre.clone(), centre.clone());
            xp[k] += h;
            xm[k] -= h;
            let fd = (objective_value(fam, 0, &xp, 0.0, obj)
                - objective_value(fam, 0, &xm, 0.0, obj))
                / (2.0 * h);
            let g = f64::from(run.grad[s * d + k]);
            assert!(
                (g - fd).abs() <= 1e-6 * (1.0 + fd.abs()),
                "row {s} coordinate {k}: grad {g} vs central difference {fd}"
            );
        }
        // The relaxation is exact, so the bound is the affine minimum.
        let lb = f64::from(run.lb[s]);
        let vmin = (0..1 << d)
            .map(|m: usize| {
                let x: Vec<f64> = (0..d)
                    .map(|k| if (m >> k) & 1 == 1 { hi[k] } else { lo[k] })
                    .collect();
                objective_value(fam, 0, &x, 0.0, obj)
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            (lb - vmin).abs() <= 1e-6 * (1.0 + vmin.abs()),
            "{lb} vs {vmin}"
        );
    }
}

fn wave(
    rng: &mut Lcg,
    d: usize,
    count: usize,
    clauses: u32,
) -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<Vec<u32>>) {
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    for _ in 0..count {
        let (l, h) = random_box(rng, d);
        lo.extend(l);
        hi.extend(h);
    }
    let clause = (0..count as u32).map(|b| b % clauses).collect();
    let rows = (0..count)
        .map(|b| match b % 3 {
            0 => vec![0],
            1 => vec![1, 0],
            _ => vec![1],
        })
        .collect();
    (lo, hi, clause, rows)
}

fn bits(r: &Run, rows: std::ops::Range<usize>, d: usize) -> Vec<u32> {
    rows.flat_map(|i| {
        let mut v = vec![r.lb[i].to_bits(), r.scale[i].to_bits()];
        v.extend(r.grad[i * d..(i + 1) * d].iter().map(|g| g.to_bits()));
        v
    })
    .collect()
}

/// A domain's outputs are the same bits whichever domains share its batch
/// and block and wherever it sits in the batch.
#[test]
fn fl_domain_results_do_not_depend_on_the_batch() {
    let mut rng = Lcg(99);
    let d = 3;
    let fam = random_family(&mut rng, d, 3, 0.03);
    let problem = problem_of(vec![fam], &mut rng, 2);
    let count = 3 * FL_LANES + 5;
    let (lo, hi, clause, rows) = wave(&mut rng, d, count, 3);
    let full = run_real::<f32>(&problem, 0, &lo, &hi, &clause, &rows, 1).expect("full");
    let mut offs = vec![0usize];
    for r in &rows {
        offs.push(offs.last().unwrap() + r.len());
    }
    // Each domain alone, and the batch reversed.
    for b in 0..count {
        let alone = run_real::<f32>(
            &problem,
            0,
            &lo[b * d..(b + 1) * d],
            &hi[b * d..(b + 1) * d],
            &clause[b..=b],
            &rows[b..=b],
            1,
        )
        .expect("alone");
        assert_eq!(
            bits(&alone, 0..rows[b].len(), d),
            bits(&full, offs[b]..offs[b + 1], d),
            "domain {b} alone differs from its batch value"
        );
    }
    let rev: Vec<usize> = (0..count).rev().collect();
    let rlo: Vec<f64> = rev
        .iter()
        .flat_map(|&b| lo[b * d..(b + 1) * d].to_vec())
        .collect();
    let rhi: Vec<f64> = rev
        .iter()
        .flat_map(|&b| hi[b * d..(b + 1) * d].to_vec())
        .collect();
    let rcl: Vec<u32> = rev.iter().map(|&b| clause[b]).collect();
    let rrows: Vec<Vec<u32>> = rev.iter().map(|&b| rows[b].clone()).collect();
    let reversed = run_real::<f32>(&problem, 0, &rlo, &rhi, &rcl, &rrows, 1).expect("reversed");
    let mut roffs = vec![0usize];
    for r in &rrows {
        roffs.push(roffs.last().unwrap() + r.len());
    }
    for (pos, &b) in rev.iter().enumerate() {
        assert_eq!(
            bits(&reversed, roffs[pos]..roffs[pos + 1], d),
            bits(&full, offs[b]..offs[b + 1], d),
            "domain {b} moved to position {pos} differs"
        );
    }
}

/// 1, 3 and 8 workers give the same bits.
#[test]
fn fl_is_thread_count_independent() {
    let mut rng = Lcg(1234);
    let d = 4;
    let fam = random_family(&mut rng, d, 2, 0.01);
    let problem = problem_of(vec![fam], &mut rng, 2);
    let count = 7 * FL_LANES + 3;
    let (lo, hi, clause, rows) = wave(&mut rng, d, count, 2);
    let base = run_real::<f32>(&problem, 0, &lo, &hi, &clause, &rows, 1).expect("1");
    let n = base.lb.len();
    for threads in [3, 8] {
        let other = run_real::<f32>(&problem, 0, &lo, &hi, &clause, &rows, threads).expect("n");
        assert_eq!(
            bits(&other, 0..n, d),
            bits(&base, 0..n, d),
            "{threads} threads"
        );
    }
    // The shipped backend routes to the same kernel.
    let backend = CpuF32Backend::with_threads(3);
    let lease = DeviceLease { bytes: 1 << 34 };
    let h = backend
        .upload_forward_linear(&problem, &lease)
        .expect("upload");
    let mut off = vec![0u32];
    let mut flat = Vec::new();
    for r in &rows {
        flat.extend_from_slice(r);
        off.push(flat.len() as u32);
    }
    let batch = FlBatch {
        family: 0,
        count,
        d,
        lo: &lo,
        hi: &hi,
        clause: &clause,
        obj_offsets: &off,
        obj_rows: &flat,
        deadline: None,
    };
    let mut out = FlOut::default();
    let stats = backend
        .forward_linear_batch(&h, &batch, &mut out)
        .expect("batch")
        .into_hint()
        .into_ingest();
    assert_eq!(stats.domains, count as u64);
    assert_eq!(stats.max_block, FL_LANES);
    let view = out.hint().into_ingest();
    for r in 0..n {
        assert_eq!(
            view.lb(r).to_bits(),
            base.lb[r].to_bits(),
            "backend row {r}"
        );
    }
}

#[test]
fn fl_refuses_unsupported_malformed_foreign_and_late_requests() {
    let mut rng = Lcg(5);
    let d = 2;
    let fam = random_family(&mut rng, d, 1, 0.0);
    let good = problem_of(vec![fam], &mut rng, 1);
    let lease = DeviceLease { bytes: 1 << 34 };
    // Mul and MaxGlobal are expressible but unsupported.
    for op in [FlOp::Mul { a: 2, b: 2 }, FlOp::MaxGlobal { src: 2 }] {
        let mut p = good.clone();
        let fam = &mut p.families[0];
        let n = fam.program.ops.len();
        fam.program.ops.push(op.clone());
        fam.program
            .sizes
            .push(if matches!(op, FlOp::Mul { .. }) { 6 } else { 1 });
        for c in &mut fam.clauses {
            c.push(FlOpConsts::None);
        }
        assert_eq!(fam.program.ops.len(), n + 1);
        assert!(p.validate().is_ok(), "{op:?} is expressible");
        assert!(
            matches!(
                fl::upload::<f32>("test", &p, &lease, 1),
                Err(Refusal::Unsupported(_))
            ),
            "{op:?} must refuse Unsupported"
        );
    }
    // A malformed problem.
    let mut bad = good.clone();
    bad.objectives[0].push(1.0);
    assert!(matches!(
        fl::upload::<f32>("test", &bad, &lease, 1),
        Err(Refusal::Invalid(_))
    ));
    // Over the lease.
    assert!(matches!(
        fl::upload::<f32>("test", &good, &DeviceLease { bytes: 16 }, 1),
        Err(Refusal::Memory { .. })
    ));
    let h = fl::upload::<f32>("test", &good, &lease, 1).expect("upload");
    let lo = vec![0.0; d];
    let hi = vec![0.1; d];
    let off = [0u32, 1];
    let rows = [0u32];
    let batch = FlBatch {
        family: 0,
        count: 1,
        d,
        lo: &lo,
        hi: &hi,
        clause: &[],
        obj_offsets: &off,
        obj_rows: &rows,
        deadline: None,
    };
    let mut out = FlOut::default();
    assert!(fl::forward_linear_batch::<f32>(&h, &batch, &mut out, 1).is_ok());
    // A malformed batch.
    let crossed = vec![0.2; d];
    assert!(matches!(
        fl::forward_linear_batch::<f32>(
            &h,
            &FlBatch {
                lo: &crossed,
                ..batch
            },
            &mut out,
            1
        ),
        Err(Refusal::Invalid(_))
    ));
    // A foreign handle (another backend's payload, or another scalar's).
    let foreign = FlHandle::new("other", &good, 3u8);
    assert!(matches!(
        fl::forward_linear_batch::<f32>(&foreign, &batch, &mut out, 1),
        Err(Refusal::Invalid(_))
    ));
    let h64 = fl::upload::<f64>("test", &good, &lease, 1).expect("upload");
    assert!(matches!(
        fl::forward_linear_batch::<f32>(&h64, &batch, &mut out, 1),
        Err(Refusal::Invalid(_))
    ));
    // A late request.
    let past = Instant::now()
        .checked_sub(Duration::from_millis(5))
        .unwrap_or_else(Instant::now);
    assert_eq!(
        fl::forward_linear_batch::<f32>(
            &h,
            &FlBatch {
                deadline: Some(past),
                ..batch
            },
            &mut out,
            1
        )
        .err(),
        Some(Refusal::Deadline)
    );
}

#[test]
fn fl_dedups_shared_weights_across_families() {
    let mut rng = Lcg(8);
    let d = 3;
    let fam = random_family(&mut rng, d, 2, 0.0);
    // Two families sharing every weight Arc (another structure's constants).
    let mut twin = fam.clone();
    twin.clauses = random_family(&mut Lcg(9), d, 3, 0.0).clauses;
    let shared = problem_of(vec![fam.clone(), twin], &mut rng, 1);
    // The same shapes with DISTINCT weight Arcs.
    let distinct = problem_of(
        vec![fam, random_family(&mut Lcg(10), d, 3, 0.0)],
        &mut rng,
        1,
    );
    let up_s = FlUpload::<f32>::lower(&shared, 1).expect("shared");
    let up_d = FlUpload::<f32>::lower(&distinct, 1).expect("distinct");
    let linears = shared.families[0]
        .program
        .ops
        .iter()
        .filter(|op| matches!(op, FlOp::Linear(_)))
        .count();
    assert_eq!(up_s.weight_count(), linears, "one upload per shared Arc");
    assert_eq!(up_d.weight_count(), 2 * linears);
    assert!(up_s.bytes() < up_d.bytes());
    // Identical constants shared by Arc upload once as well: two clauses that
    // share every constant Arc cost exactly what one clause costs.
    let c0 = shared.families[0].clauses[0].clone();
    let program = shared.families[0].program.clone();
    let of = |clauses: Vec<Vec<FlOpConsts>>| FlProblem {
        families: vec![FlFamily {
            program: program.clone(),
            clauses,
        }],
        objectives: vec![vec![1.0, 0.0]],
    };
    let two = FlUpload::<f32>::lower(&of(vec![c0.clone(), c0.clone()]), 1).expect("two");
    let one = FlUpload::<f32>::lower(&of(vec![c0]), 1).expect("one");
    assert_eq!(two.bytes(), one.bytes(), "shared constants upload once");
}

#[test]
fn fl_work_bytes_cover_the_workspace() {
    for (d, sizes, sig) in [
        (1usize, vec![1usize, 1], vec![false, false]),
        (4, vec![4, 64, 64, 2], vec![false, false, true, false]),
        (
            16,
            vec![16, 300, 300, 17, 1],
            vec![false, true, false, true, false],
        ),
    ] {
        let est = fl::work_bytes::<f32>(d, &sizes, &sig);
        let real = fl::workspace_alloc_bytes::<f32>(d, &sizes, &sig);
        assert!(est >= real, "d={d} sizes={sizes:?}: {est} < {real}");
        let est64 = fl::work_bytes::<f64>(d, &sizes, &sig);
        let real64 = fl::workspace_alloc_bytes::<f64>(d, &sizes, &sig);
        assert!(est64 >= real64);
        assert!(est64 >= est);
    }
}

/// ROLE PROBE (not a gate): certified-free search throughput of the CPU f32
/// backend on an nn4sys-shaped trunk, run by the integrator under `ny_box.sh
/// measure`. A throughput measurement, so it lives behind the off-by-default
/// `benchmarks` feature rather than `#[ignore]`: `targo --unverified test
/// --release --locked --offline -p ny-bbe --features benchmarks --lib
/// fl_cpu_rate_probe -- --nocapture`.
#[cfg(feature = "benchmarks")]
#[test]
fn fl_cpu_rate_probe() {
    let mut rng = Lcg(31337);
    let d = 4;
    let widths = [d, 512, 512, 512, 512, 1];
    let mut ops = vec![FlOp::Input];
    let mut sizes = vec![d];
    let mut consts = vec![FlOpConsts::None];
    let mut prev = 0usize;
    for (i, &w) in widths.iter().enumerate().skip(1) {
        let n_in = sizes[prev];
        ops.push(lin(
            prev,
            n_in,
            w,
            weights(&mut rng, w, n_in, 1.0 / (n_in as f64).sqrt()),
        ));
        sizes.push(w);
        consts.push(bias(&mut rng, w, 0.0, 0.0));
        prev = sizes.len() - 1;
        if i + 1 < widths.len() {
            ops.push(FlOp::Relu { src: prev as u32 });
            sizes.push(w);
            consts.push(FlOpConsts::None);
            prev = sizes.len() - 1;
        }
    }
    let seed = sizes.len() - 1;
    let problem = FlProblem {
        families: vec![FlFamily {
            program: FlProgram {
                d,
                sizes,
                seed,
                ops,
            },
            clauses: vec![consts],
        }],
        objectives: vec![vec![1.0]],
    };
    let count = 4096;
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    for _ in 0..count {
        let (l, h) = random_box(&mut rng, d);
        lo.extend(l);
        hi.extend(h);
    }
    let rows = vec![vec![0u32]; count];
    let t0 = Instant::now();
    let run = run_real::<f32>(&problem, 0, &lo, &hi, &[], &rows, 0).expect("runs");
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(run.lb.len(), count);
    println!(
        "[fl-rate] domains={count} secs={secs:.3} domains_per_s={:.0} lanes={FL_LANES}",
        count as f64 / secs
    );
}
