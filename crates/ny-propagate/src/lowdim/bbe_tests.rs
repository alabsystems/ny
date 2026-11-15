// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M8a (`#m8-input-split-batch`) lane tests: the untrusted forward-linear
//! wave search (`bbe`) and the same-box disjunction tree (`disjunction`).
//!
//! The backends here are written in the test (ny-propagate cannot
//! dev-depend on ny-bbe): an exact REFERENCE that returns the lane's own f64
//! bounds rounded to f32, and GARBAGE backends (claim everything, never
//! claim, NaN, random, flipped gradients, permuted domains, anti-claims,
//! lies, bad shapes, refusals and panics), most of them honest on the probe
//! so the searched path is really exercised. The load-bearing tests are the
//! SAT MOATS (a planted violation is never verified whatever the backend
//! returns), CONTAINMENT (every bound a close reads is at or below sampled
//! truth), the VOID GUARDS (the searched path really batched and certified),
//! LEGACY BIT-IDENTITY (golden counters recorded at `ce4f09170`) and
//! thread-count independence.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::search::{
    BackendCaps, DeviceLease, FlBatch, FlHandle, FlOp, FlOpConsts, FlOut, FlProblem, FlStats,
    Refusal, SearchBackend, SearchPrecision, Untrusted,
};
use ny_tensor::BoundedTensor;

use super::bbe::{self, ClaimMark};
use super::disjunction::{
    disjunction_admits, prove_disjunction_admitted, prove_disjunction_admitted_with,
    DisjunctionClause,
};
use super::search::{self, Fate, Wave};
use super::shared;
use super::*;
use crate::layers::{ConcatLayer, LinearLayer, ReLULayer, SliceLayer};
use crate::network::GraphNode;

// =====================================================================
//  FIXTURES (verbatim in the ce4f09170 golden probe; do not edit without
//  re-recording the goldens there)
// =====================================================================

/// Deterministic LCG in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn lin(rng: &mut Lcg, n_out: usize, n_in: usize, scale: f64) -> Layer {
    let w = Array2::from_shape_fn((n_out, n_in), |_| (rng.next() * scale) as f32);
    let b = Array1::from_shape_fn(n_out, |_| (rng.next() * 0.3) as f32);
    Layer::Linear(LinearLayer::new(w, Some(b)).expect("linear"))
}

/// `d -> width -> ReLU -> width -> ReLU -> 2`.
fn relu_net(seed: u64, d: usize, width: usize) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        lin(&mut rng, width, d, 1.2),
        vec![NETWORK_INPUT.to_string()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        lin(&mut rng, width, width, 0.7),
        vec!["r1".into()],
    ));
    g.add_node(GraphNode::new(
        "r2",
        Layer::ReLU(ReLULayer::new()),
        vec!["l2".into()],
    ));
    g.add_node(GraphNode::new(
        "l3",
        lin(&mut rng, 2, width, 0.8),
        vec!["r2".into()],
    ));
    g.set_output("l3");
    g
}

/// linearizenn-shaped: a ReLU trunk, then an affine head that re-reads the
/// input through a Slice and a Concat (`y0 = a - b`, `y1 = b - a`).
fn lz_net(seed: u64, d: usize, width: usize) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    let inp = NETWORK_INPUT.to_string();
    g.add_node(GraphNode::new(
        "l1",
        lin(&mut rng, width, d, 1.5),
        vec![inp.clone()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        lin(&mut rng, 2, width, 0.6),
        vec!["r1".into()],
    ));
    let w = |rows: Vec<Vec<f32>>| {
        let (r, c) = (rows.len(), rows[0].len());
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        Layer::Linear(
            LinearLayer::new(Array2::from_shape_vec((r, c), flat).expect("shape"), None)
                .expect("linear"),
        )
    };
    g.add_node(GraphNode::new(
        "a",
        w(vec![vec![-0.74, -0.44]]),
        vec!["l2".into()],
    ));
    g.add_node(GraphNode::new(
        "sl",
        Layer::Slice(SliceLayer::new(1, 1, d)),
        vec![inp],
    ));
    let w2: Vec<f32> = (1..d).map(|i| -1.5 - i as f32).collect();
    g.add_node(GraphNode::new("b", w(vec![w2]), vec!["sl".into()]));
    g.add_node(GraphNode::new(
        "cat",
        Layer::Concat(ConcatLayer::new(1)),
        vec!["a".into(), "b".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        w(vec![vec![1.0, -1.0], vec![-1.0, 1.0]]),
        vec!["cat".into()],
    ));
    g.set_output("out");
    g
}

fn boxed(lo: &[f64], hi: &[f64]) -> BoundedTensor {
    let d = lo.len();
    let l = ArrayD::from_shape_vec(IxDyn(&[1, d]), lo.iter().map(|&v| v as f32).collect())
        .expect("shape");
    let h = ArrayD::from_shape_vec(IxDyn(&[1, d]), hi.iter().map(|&v| v as f32).collect())
        .expect("shape");
    BoundedTensor::new(l, h).expect("box")
}

/// f32-representable root box.
fn root_box(d: usize) -> (Vec<f64>, Vec<f64>) {
    let lo = (0..d)
        .map(|i| f64::from(-0.75f32 + 0.125 * i as f32))
        .collect();
    let hi = (0..d)
        .map(|i| f64::from(0.5f32 + 0.25 * i as f32))
        .collect();
    (lo, hi)
}

fn sample(rng: &mut Lcg, lo: &[f64], hi: &[f64]) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(l, h)| l + (h - l) * 0.5 * (rng.next() + 1.0))
        .collect()
}

/// `(min, max)` of `objective . y` over `n` samples (and the box corners).
fn sampled_range(prog: &Program, lo: &[f64], hi: &[f64], o: &[f64], seed: u64) -> (f64, f64) {
    let mut rng = Lcg(seed);
    let d = lo.len();
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..(4000 + (1 << d)) {
        let x: Vec<f64> = if i < (1 << d) {
            (0..d)
                .map(|k| if (i >> k) & 1 == 1 { hi[k] } else { lo[k] })
                .collect()
        } else {
            sample(&mut rng, lo, hi)
        };
        let v = point_objective(prog, &x, o);
        mn = mn.min(v);
        mx = mx.max(v);
    }
    (mn, mx)
}

/// One single-conjunction fixture.
struct Fx {
    prog: Program,
    lo: Vec<f64>,
    hi: Vec<f64>,
    objs: Vec<Vec<f64>>,
    thrs: Vec<f64>,
    cfg: LowDimConfig,
}

/// F1 (true margin), F2 (two conjuncts, one impossible everywhere; gradient
/// split), F3 (a violated threshold; clip + sbwr, stops on a candidate).
fn core_fixtures() -> Vec<Fx> {
    let mut out = Vec::new();
    {
        let d = 3;
        let g = relu_net(11, d, 16);
        let (lo, hi) = root_box(d);
        let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
        let o = vec![1.0, -1.0];
        let (mn, mx) = sampled_range(&prog, &lo, &hi, &o, 5);
        out.push(Fx {
            prog,
            lo,
            hi,
            objs: vec![o],
            thrs: vec![mn - 0.1 * (mx - mn)],
            cfg: LowDimConfig {
                batch: 64,
                chunk: 8,
                ..LowDimConfig::default()
            },
        });
    }
    {
        let d = 4;
        let g = lz_net(12, d, 24);
        let (lo, hi) = root_box(d);
        let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
        let (o0, o1) = (vec![1.0, 0.0], vec![0.0, 1.0]);
        let (mn0, mx0) = sampled_range(&prog, &lo, &hi, &o0, 6);
        let (_, mx1) = sampled_range(&prog, &lo, &hi, &o1, 7);
        out.push(Fx {
            prog,
            lo,
            hi,
            objs: vec![o0, o1],
            thrs: vec![mn0 - 0.05 * (mx0 - mn0), mx1 + 1.0],
            cfg: LowDimConfig {
                batch: 32,
                chunk: 4,
                split: LowDimSplit::Gradient,
                ..LowDimConfig::default()
            },
        });
    }
    {
        let d = 2;
        let g = relu_net(13, d, 12);
        let (lo, hi) = root_box(d);
        let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
        let o = vec![0.0, 1.0];
        let (mn, mx) = sampled_range(&prog, &lo, &hi, &o, 8);
        out.push(Fx {
            prog,
            lo,
            hi,
            objs: vec![o],
            thrs: vec![mn + 0.1 * (mx - mn)],
            cfg: LowDimConfig {
                batch: 16,
                chunk: 4,
                clip: true,
                split: LowDimSplit::SbWr,
                ..LowDimConfig::default()
            },
        });
    }
    out
}

fn run_core(fx: &Fx, cfg: &LowDimConfig, secs: u64) -> (LowDimOutcome, Option<Vec<f64>>) {
    let mut witness = None;
    let o = prove_conjunction_core(
        fx.prog.clone(),
        fx.lo.clone(),
        fx.hi.clone(),
        fx.objs.clone(),
        fx.thrs.clone(),
        Instant::now() + Duration::from_secs(secs),
        cfg,
        &mut witness,
    );
    (o, witness)
}

/// Families for the shared-worklist / family-core entries: two structures,
/// clauses over sub-boxes of the root with true and violated thresholds.
fn family_fixtures() -> (Vec<Vec<FamClause>>, Vec<Vec<bool>>) {
    let mut fams = Vec::new();
    let mut truth = Vec::new();
    for (seed, n) in [(21u64, 3usize), (22, 2)] {
        let d = 3;
        let g = relu_net(seed, d, 10);
        let (lo0, hi0) = root_box(d);
        let prog = lower_graph(&g, &boxed(&lo0, &hi0)).expect("lowers");
        let mut fam = Vec::new();
        let mut tr = Vec::new();
        for c in 0..n {
            let t = c as f64 / (n as f64);
            let lo: Vec<f64> = lo0
                .iter()
                .zip(&hi0)
                .map(|(l, h)| l + (h - l) * 0.2 * t)
                .collect();
            let hi: Vec<f64> = lo0
                .iter()
                .zip(&hi0)
                .map(|(l, h)| h - (h - l) * 0.2 * (1.0 - t))
                .collect();
            let o = if c % 2 == 0 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, -1.0]
            };
            let (mn, mx) = sampled_range(&prog, &lo, &hi, &o, 30 + c as u64);
            let holds = c != 1;
            let thr = if holds {
                mn - 0.08 * (mx - mn)
            } else {
                mn + 0.05 * (mx - mn)
            };
            fam.push(FamClause {
                prog: prog.clone(),
                lo,
                hi,
                objs: vec![o],
                thrs: vec![thr],
                combine: Combine::Each,
            });
            tr.push(holds);
        }
        fams.push(fam);
        truth.push(tr);
    }
    (fams, truth)
}

fn fam_cfg() -> LowDimConfig {
    LowDimConfig {
        batch: 32,
        chunk: 4,
        ..LowDimConfig::default()
    }
}

// =====================================================================
//  Helpers
// =====================================================================

/// `(kind, stop, domains, closed, max_depth, waves)`.
fn counters(o: &LowDimOutcome) -> (&'static str, Option<LowDimStop>, u64, u64, u32, u64) {
    match o {
        LowDimOutcome::Verified(s) => ("verified", None, s.domains, s.closed, s.max_depth, s.waves),
        LowDimOutcome::Unproved(s, stop) => (
            "unproved",
            Some(*stop),
            s.domains,
            s.closed,
            s.max_depth,
            s.waves,
        ),
        LowDimOutcome::Declined(_) => ("declined", None, 0, 0, 0, 0),
    }
}

fn verified(o: &LowDimOutcome) -> bool {
    matches!(o, LowDimOutcome::Verified(_))
}

fn stats_of(o: &LowDimOutcome) -> LowDimStats {
    match o {
        LowDimOutcome::Verified(s) | LowDimOutcome::Unproved(s, _) => *s,
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
}

fn bbe_cfg(base: &LowDimConfig) -> LowDimConfig {
    LowDimConfig {
        batch_engine: LowDimBatchEngine::Bbe,
        ..base.clone()
    }
}

/// A field of a `[bbe]` line.
fn field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split_whitespace()
        .find_map(|t| t.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("no {key}= in {line}"))
}

fn field_u64(line: &str, key: &str) -> u64 {
    field(line, key).parse().expect("integer field")
}

/// The test backends.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    /// The lane's own f64 bounds, rounded to f32.
    Reference,
    /// Every row a huge finite bound: claims everything.
    ClaimAll,
    /// Every row a hugely negative bound: claims nothing.
    NeverClaim,
    /// NaN everywhere.
    Nan,
    /// Random bounds and gradients.
    Random,
    /// Honest bounds, negated gradients.
    FlipGrads,
    /// Honest values of ANOTHER domain (the wave reversed).
    Permute,
    /// Claims exactly the rows the lane's f64 bound will NOT close.
    Anti,
    /// Honest plus a large offset (fails the probe).
    Lying,
    /// An output of the wrong shape.
    BadShape,
    /// `Refusal::Deadline`.
    Deadline,
    /// `Refusal::Invalid`.
    RefuseInvalid,
    /// `Refusal::Memory` from the batch.
    RefuseMemory,
    /// A panic in the batch.
    Panic,
    /// `Refusal::Memory` from the upload.
    UploadMemory,
    /// `Refusal::Unsupported` from the upload.
    UploadUnsupported,
    /// A panic in the upload.
    UploadPanic,
    /// Panics on ANY call (the legacy path must never consult it).
    PanicEverywhere,
}

struct TbUpload {
    objectives: Vec<Vec<f64>>,
}

struct TestBackend {
    lane: Vec<Vec<Program>>,
    thr_of_row: Vec<f64>,
    mode: Mode,
    honest_calls: usize,
    calls: AtomicUsize,
}

impl TestBackend {
    fn new(lane: Vec<Vec<Program>>, mode: Mode, honest_calls: usize) -> Arc<Self> {
        Arc::new(Self {
            lane,
            thr_of_row: Vec::new(),
            mode,
            honest_calls,
            calls: AtomicUsize::new(0),
        })
    }

    fn with_thresholds(mut self: Arc<Self>, thr: Vec<f64>) -> Arc<Self> {
        Arc::get_mut(&mut self).expect("unique").thr_of_row = thr;
        self
    }

    fn single(fx: &Fx, mode: Mode, honest_calls: usize) -> Arc<Self> {
        Self::new(vec![vec![fx.prog.clone()]], mode, honest_calls).with_thresholds(fx.thrs.clone())
    }
}

impl SearchBackend for TestBackend {
    fn name(&self) -> &'static str {
        "test"
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps::Cpu {
            threads: 1,
            lanes: 1,
            precision: SearchPrecision::F64,
        }
    }

    fn upload_forward_linear(
        &self,
        problem: &FlProblem,
        _lease: &DeviceLease,
    ) -> Result<FlHandle, Refusal> {
        match self.mode {
            Mode::UploadMemory => return Err(Refusal::Memory { need: 2, cap: 1 }),
            Mode::UploadUnsupported => return Err(Refusal::Unsupported("test")),
            Mode::UploadPanic | Mode::PanicEverywhere => panic!("test backend upload panic"),
            _ => {}
        }
        problem.validate().map_err(Refusal::Invalid)?;
        Ok(FlHandle::new(
            "test",
            problem,
            TbUpload {
                objectives: problem.objectives.clone(),
            },
        ))
    }

    fn forward_linear_batch(
        &self,
        handle: &FlHandle,
        batch: &FlBatch<'_>,
        out: &mut FlOut,
    ) -> Result<Untrusted<FlStats>, Refusal> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(
            self.mode != Mode::PanicEverywhere,
            "test backend batch panic"
        );
        let up = handle
            .payload::<TbUpload>()
            .ok_or_else(|| Refusal::Invalid("foreign".into()))?;
        batch.validate(handle).map_err(Refusal::Invalid)?;
        let honest = n < self.honest_calls || self.mode == Mode::Reference;
        if !honest {
            match self.mode {
                Mode::Deadline => return Err(Refusal::Deadline),
                Mode::RefuseInvalid => return Err(Refusal::Invalid("test".into())),
                Mode::RefuseMemory => return Err(Refusal::Memory { need: 2, cap: 1 }),
                Mode::Panic => panic!("test backend batch panic"),
                _ => {}
            }
        }
        let d = batch.d;
        let rows = batch.obj_rows.len();
        let progs: Vec<&Program> = self.lane[batch.family as usize].iter().collect();
        // The lane's own f64 bound per domain (chunk of one).
        let mut lb = Vec::with_capacity(rows);
        let mut scale = Vec::with_capacity(rows);
        let mut grad = Vec::with_capacity(rows * d);
        for b in 0..batch.count {
            let c = batch.clause.get(b).map_or(0, |&c| c as usize);
            let r0 = batch.obj_offsets[b] as usize;
            let r1 = batch.obj_offsets[b + 1] as usize;
            let objs: Vec<Vec<f64>> = batch.obj_rows[r0..r1]
                .iter()
                .map(|&r| up.objectives[r as usize].clone())
                .collect();
            let (l, h) = (&batch.lo[b * d..(b + 1) * d], &batch.hi[b * d..(b + 1) * d]);
            let db = shared::bound_group(&progs, &[c], l, h, &[objs.as_slice()], 1, false)
                .pop()
                .expect("one bound");
            for i in 0..objs.len() {
                let g = &db.grads[i * d..(i + 1) * d];
                let s = db.lbs[i].abs()
                    + g.iter()
                        .zip(l.iter().zip(h))
                        .map(|(g, (a, b))| g.abs() * a.abs().max(b.abs()))
                        .sum::<f64>();
                lb.push(db.lbs[i]);
                scale.push(s);
                grad.extend_from_slice(g);
            }
        }
        if !honest {
            match self.mode {
                Mode::ClaimAll => {
                    lb.iter_mut().for_each(|v| *v = 1e30);
                    scale.iter_mut().for_each(|v| *v = 0.0);
                }
                Mode::NeverClaim => lb.iter_mut().for_each(|v| *v = -1e30),
                Mode::Nan => {
                    lb.iter_mut().for_each(|v| *v = f64::NAN);
                    scale.iter_mut().for_each(|v| *v = f64::NAN);
                    grad.iter_mut().for_each(|v| *v = f64::NAN);
                }
                Mode::Random => {
                    let mut rng = Lcg(0x9e37_79b9 ^ n as u64);
                    lb.iter_mut().for_each(|v| *v = rng.next() * 10.0);
                    grad.iter_mut().for_each(|v| *v = rng.next() * 3.0);
                }
                Mode::FlipGrads => grad.iter_mut().for_each(|v| *v = -*v),
                Mode::Permute => {
                    let per = rows / batch.count.max(1);
                    if per * batch.count == rows {
                        let (l0, s0, g0) = (lb.clone(), scale.clone(), grad.clone());
                        for b in 0..batch.count {
                            let src = batch.count - 1 - b;
                            for i in 0..per {
                                let (r, q) = (b * per + i, src * per + i);
                                lb[r] = l0[q];
                                scale[r] = s0[q];
                                grad[r * d..(r + 1) * d].copy_from_slice(&g0[q * d..(q + 1) * d]);
                            }
                        }
                    }
                }
                Mode::Anti => {
                    for (r, v) in lb.iter_mut().enumerate() {
                        let row = batch.obj_rows[r] as usize;
                        let thr = self.thr_of_row.get(row).copied().unwrap_or(0.0);
                        *v = if *v > thr { -1e30 } else { 1e30 };
                    }
                    scale.iter_mut().for_each(|v| *v = 0.0);
                }
                Mode::Lying => lb.iter_mut().for_each(|v| *v += 5.0),
                Mode::BadShape => {
                    out.prepare(rows + 1, d);
                    return Ok(Untrusted::new(FlStats::default()));
                }
                _ => {}
            }
        }
        out.prepare(rows, d);
        let w = out.writer();
        for r in 0..rows {
            w.lb[r] = lb[r] as f32;
            w.scale[r] = scale[r] as f32;
            for k in 0..d {
                w.grad[r * d + k] = grad[r * d + k] as f32;
            }
        }
        Ok(Untrusted::new(FlStats {
            domains: batch.count as u64,
            rows: rows as u64,
            ..FlStats::default()
        }))
    }
}

fn with_backend<T>(b: &Arc<TestBackend>, f: impl FnOnce() -> T) -> T {
    let dynb: Arc<dyn SearchBackend> = Arc::clone(b) as Arc<dyn SearchBackend>;
    bbe::with_test_backend(dynb, f)
}

// =====================================================================
//  Legacy bit-identity (goldens recorded at ce4f09170)
// =====================================================================

/// `(kind, stop, domains, closed, max_depth, waves)` of F1..F3 on the
/// legacy lane at `ce4f09170`, recorded by a probe (this file's FIXTURES
/// section verbatim) in a detached worktree of that commit; the probe is not
/// committed.
const GOLDEN_CORE: [(&str, Option<LowDimStop>, u64, u64, u32, u64); 3] = [
    ("verified", None, 15, 8, 3, 4),
    ("verified", None, 13, 7, 6, 7),
    ("unproved", Some(LowDimStop::Candidate), 3, 0, 2, 2),
];

/// F3's candidate witness at `ce4f09170` (bits).
const GOLDEN_F3_WITNESS: [u64; 2] = [4_595_923_419_731_591_168, 13_809_132_590_559_199_792];

/// Per family, per clause `(kind, stop, domains, closed, max_depth)` of the
/// shared worklist and of the family core at `ce4f09170` (family 1 has two
/// clauses; its third row is padding).
type FamGolden = [[(&'static str, Option<LowDimStop>, u64, u64, u32); 3]; 2];
const GOLDEN_SHARED: FamGolden = [
    [
        ("verified", None, 19, 10, 5),
        ("unproved", Some(LowDimStop::Candidate), 90, 38, 12),
        ("verified", None, 27, 14, 6),
    ],
    [
        ("verified", None, 29, 15, 5),
        ("unproved", Some(LowDimStop::Candidate), 127, 42, 12),
        ("", None, 0, 0, 0),
    ],
];
const GOLDEN_FAMILY_CORE: FamGolden = [
    [
        ("verified", None, 19, 10, 5),
        ("unproved", Some(LowDimStop::Candidate), 98, 43, 12),
        ("verified", None, 27, 14, 6),
    ],
    [
        ("verified", None, 29, 15, 5),
        ("unproved", Some(LowDimStop::Candidate), 155, 53, 12),
        ("", None, 0, 0, 0),
    ],
];

fn assert_core_goldens(tag: &str, runs: &[(LowDimOutcome, Option<Vec<f64>>)]) {
    for (i, ((o, w), g)) in runs.iter().zip(GOLDEN_CORE).enumerate() {
        let (kind, stop, dom, closed, depth, waves) = counters(o);
        assert_eq!(
            (kind, stop, dom, closed, depth, waves),
            g,
            "{tag}: fixture F{} differs from its ce4f09170 golden",
            i + 1
        );
        if i == 2 {
            let bits: Vec<u64> = w
                .as_ref()
                .expect("F3 witness")
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(bits, GOLDEN_F3_WITNESS.to_vec(), "{tag}: F3 witness bits");
        }
    }
}

fn assert_fam_goldens(tag: &str, outs: &[Vec<LowDimOutcome>], golden: &FamGolden) {
    for (f, fam) in outs.iter().enumerate() {
        for (c, o) in fam.iter().enumerate() {
            let (kind, stop, dom, closed, depth, _) = counters(o);
            assert_eq!(
                (kind, stop, dom, closed, depth),
                golden[f][c],
                "{tag}: family {f} clause {c} differs from its ce4f09170 golden"
            );
        }
    }
}

#[test]
fn lowdim_batch_engine_legacy_is_bit_identical_and_never_consults_the_backend() {
    assert_eq!(
        LowDimConfig::default().batch_engine,
        LowDimBatchEngine::Legacy
    );
    assert_eq!(LowDimConfig::default().engine, InputSplitEngine::Legacy);
    let fx = core_fixtures();
    // A backend that panics on ANY call is injected: legacy never consults it.
    let panicky = TestBackend::new(Vec::new(), Mode::PanicEverywhere, 0);
    let _ = take_last_bbe_summary();
    let runs: Vec<(LowDimOutcome, Option<Vec<f64>>)> = with_backend(&panicky, || {
        fx.iter().map(|f| run_core(f, &f.cfg, 60)).collect()
    });
    assert_eq!(panicky.calls.load(Ordering::SeqCst), 0);
    assert!(
        take_last_bbe_summary().is_none(),
        "the legacy lane printed a [bbe] line"
    );
    assert_core_goldens("legacy", &runs);
}

#[test]
fn input_split_engine_legacy_is_bit_identical_on_every_entry() {
    let panicky = TestBackend::new(Vec::new(), Mode::PanicEverywhere, 0);
    let explicit = |c: &LowDimConfig| LowDimConfig {
        engine: InputSplitEngine::Legacy,
        batch_engine: LowDimBatchEngine::Legacy,
        ..c.clone()
    };
    with_backend(&panicky, || {
        // The single-conjunction core (the grouped and singleton entries'
        // per-clause lane).
        let fx = core_fixtures();
        let runs: Vec<_> = fx
            .iter()
            .map(|f| run_core(f, &explicit(&f.cfg), 60))
            .collect();
        assert_core_goldens("engine=legacy core", &runs);
        // The fold's two schedulers.
        let deadline = Instant::now() + Duration::from_secs(60);
        let (fams, _) = family_fixtures();
        let (shared_out, _) = shared::prove_families_shared(fams, deadline, &explicit(&fam_cfg()));
        assert_fam_goldens("engine=legacy shared", &shared_out, &GOLDEN_SHARED);
        let (fams, _) = family_fixtures();
        let core_out: Vec<Vec<LowDimOutcome>> = fams
            .into_iter()
            .map(|fam| prove_family_core(fam, deadline, &explicit(&fam_cfg())))
            .collect();
        assert_fam_goldens("engine=legacy family core", &core_out, &GOLDEN_FAMILY_CORE);
    });
    assert_eq!(panicky.calls.load(Ordering::SeqCst), 0);
    assert!(!fold_shared(&LowDimConfig::default()));
    assert!(fold_shared(&LowDimConfig {
        engine: InputSplitEngine::Bbe,
        ..LowDimConfig::default()
    }));
    assert!(fold_shared(&LowDimConfig {
        shared_worklist: true,
        ..LowDimConfig::default()
    }));
}

#[test]
fn lowdim_bbe_without_a_registered_backend_runs_legacy_bit_identical() {
    assert!(
        !forward_linear_search_backend_registered(),
        "no test in this binary may register a process factory"
    );
    let fx = core_fixtures();
    let runs: Vec<_> = fx
        .iter()
        .map(|f| {
            let r = run_core(f, &bbe_cfg(&f.cfg), 60);
            let line = take_last_bbe_summary().expect("an armed run prints its line");
            // F3 arms the clip, which the admission refuses first.
            let why = if f.cfg.clip {
                "unsupported:clip"
            } else {
                "no_backend"
            };
            assert_eq!(field(&line, "fallback"), why, "{line}");
            assert_eq!(field(&line, "probe"), "skip", "{line}");
            r
        })
        .collect();
    assert_core_goldens("bbe without a backend", &runs);
}

#[test]
fn lowdim_bbe_refusals_panics_and_bad_shapes_fall_back_to_legacy() {
    let fx = core_fixtures();
    let legacy: Vec<_> = fx.iter().map(|f| run_core(f, &f.cfg, 60)).collect();
    for (mode, reason) in [
        (Mode::UploadMemory, "memory"),
        (Mode::UploadUnsupported, "unsupported:backend"),
        (Mode::UploadPanic, "panic"),
        (Mode::RefuseInvalid, "invalid"),
        (Mode::RefuseMemory, "memory"),
        (Mode::Panic, "panic_or_bad_shape"),
        (Mode::BadShape, "panic_or_bad_shape"),
    ] {
        // F1 and F2 (F3 arms the clip, which the admission refuses first).
        for (f, l) in fx.iter().zip(&legacy).take(2) {
            // Honest on the probe (call 0), then the failure on the first wave.
            let b = TestBackend::single(f, mode, 1);
            let (o, w) = with_backend(&b, || run_core(f, &bbe_cfg(&f.cfg), 60));
            let line = take_last_bbe_summary().expect("line");
            assert_eq!(field(&line, "fallback"), reason, "{mode:?}: {line}");
            assert_eq!(
                counters(&o),
                counters(&l.0),
                "{mode:?}: the fallback is legacy"
            );
            assert_eq!(w, l.1, "{mode:?}: witness");
            assert_eq!(field_u64(&line, "claimed"), 0, "{mode:?}: {line}");
        }
    }
}

#[test]
fn lowdim_bbe_deadline_refusal_stops_unproved() {
    let fx = core_fixtures();
    let f = &fx[0];
    let b = TestBackend::single(f, Mode::Deadline, 1);
    let (o, _) = with_backend(&b, || run_core(f, &bbe_cfg(&f.cfg), 60));
    match o {
        LowDimOutcome::Unproved(_, stop) => assert_eq!(stop, LowDimStop::Deadline),
        other => panic!("a deadline refusal must stop the lane unproved: {other:?}"),
    }
    let line = take_last_bbe_summary().expect("line");
    assert_eq!(
        field(&line, "fallback"),
        "none",
        "a deadline is not a latch: {line}"
    );
}

#[test]
fn lowdim_bbe_probe_rejects_a_lying_backend() {
    let fx = core_fixtures();
    for f in fx.iter().take(2) {
        let legacy = run_core(f, &f.cfg, 60);
        for mode in [Mode::Lying, Mode::Nan] {
            // Lying from the FIRST call: the probe itself sees the lie.
            let b = TestBackend::single(f, mode, 0);
            let (o, w) = with_backend(&b, || run_core(f, &bbe_cfg(&f.cfg), 60));
            let line = take_last_bbe_summary().expect("line");
            assert_eq!(field(&line, "probe"), "fail", "{mode:?}: {line}");
            assert_eq!(field(&line, "fallback"), "probe", "{mode:?}: {line}");
            assert_eq!(counters(&o), counters(&legacy.0));
            assert_eq!(w, legacy.1);
            assert_eq!(b.calls.load(Ordering::SeqCst), 1, "only the probe call");
        }
    }
}

#[test]
fn lowdim_bbe_admission_refuses_refine_joint_alpha_clip_mul_and_maxglobal() {
    let fx = core_fixtures();
    let p = &fx[0].prog;
    let base = LowDimConfig::default();
    assert!(bbe::bbe_admits(&[p], &base).is_ok());
    for (cfg, why) in [
        (
            LowDimConfig {
                refine_max_width: 4,
                ..base.clone()
            },
            "refine",
        ),
        (
            LowDimConfig {
                joint_quadratic: true,
                ..base.clone()
            },
            "joint",
        ),
        (
            LowDimConfig {
                alpha_iters: 5,
                ..base.clone()
            },
            "alpha",
        ),
        (LowDimConfig { clip: true, ..base }, "clip"),
    ] {
        assert_eq!(bbe::bbe_admits(&[p], &cfg), Err(why));
    }
    let mut with_mul = p.clone();
    with_mul.ops.push(Op::Mul { a: 1, b: 1 });
    with_mul.sizes.push(with_mul.sizes[1]);
    assert_eq!(bbe::bbe_admits(&[&with_mul], &base), Err("mul"));
    let mut with_max = p.clone();
    with_max.ops.push(Op::MaxGlobal { src: 1 });
    with_max.sizes.push(1);
    assert_eq!(bbe::bbe_admits(&[&with_max], &base), Err("maxglobal"));
    // A refused run is the legacy run, and says why (F3 arms the clip).
    let f = &fx[2];
    let b = TestBackend::single(f, Mode::Reference, 0);
    let legacy = run_core(f, &f.cfg, 60);
    let (o, _) = with_backend(&b, || run_core(f, &bbe_cfg(&f.cfg), 60));
    let line = take_last_bbe_summary().expect("line");
    assert_eq!(field(&line, "fallback"), "unsupported:clip", "{line}");
    assert_eq!(counters(&o), counters(&legacy.0));
    assert_eq!(b.calls.load(Ordering::SeqCst), 0);
}

/// Random sub-boxes of the root: the f64 bound every claim is certified
/// with (claimed domains in domain order, cut at the run's chunk) is at or
/// below sampled truth, and every CLOSED fate's box holds no point that
/// violates all conjuncts.
#[test]
fn lowdim_bbe_certified_bounds_never_exceed_a_sampled_value() {
    let fx = core_fixtures();
    let mut checked = 0usize;
    for f in fx.iter().take(2) {
        let d = f.prog.d;
        let cfg = bbe_cfg(&f.cfg);
        let w0: Vec<f64> = f.lo.iter().zip(&f.hi).map(|(l, h)| h - l).collect();
        let wave = Wave {
            prog: &f.prog,
            objs: &f.objs,
            thrs: &f.thrs,
            cfg: &cfg,
            w0: &w0,
            chunk: cfg.chunk,
        };
        let mut rng = Lcg(404);
        let count = 40;
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for _ in 0..count {
            let a = sample(&mut rng, &f.lo, &f.hi);
            let b = sample(&mut rng, &f.lo, &f.hi);
            let s = 0.1 + 0.4 * (rng.next() + 1.0);
            for k in 0..d {
                let (l, h) = (a[k].min(b[k]), a[k].max(b[k]));
                let m = f64::midpoint(l, h);
                lo.push(m - s * (m - l));
                hi.push(m + s * (h - m));
            }
        }
        let b = TestBackend::single(f, Mode::Reference, 0);
        let fates = with_backend(&b, || {
            let mut run = bbe::arm_single(&f.prog, &f.lo, &f.hi, &f.objs, &cfg);
            assert!(run.live(), "the reference backend passes the probe");
            match bbe::single_wave(
                &mut run,
                &wave,
                &lo,
                &hi,
                count,
                Instant::now() + Duration::from_secs(60),
            ) {
                bbe::WaveResult::Done(fates) => fates,
                _ => panic!("the reference wave must complete"),
            }
        });
        // The certify arrangement: claimed domains in order, cut at chunk.
        let claimed: Vec<usize> = (0..count)
            .filter(|&b| fates[b].1.claim != ClaimMark::Unclaimed)
            .collect();
        assert!(!claimed.is_empty(), "VOID: nothing claimed");
        let clo: Vec<f64> = claimed
            .iter()
            .flat_map(|&b| lo[b * d..(b + 1) * d].to_vec())
            .collect();
        let chi: Vec<f64> = claimed
            .iter()
            .flat_map(|&b| hi[b * d..(b + 1) * d].to_vec())
            .collect();
        for s in (0..claimed.len()).step_by(cfg.chunk) {
            let m = cfg.chunk.min(claimed.len() - s);
            let ch = Chunk::new(m, d, &clo[s * d..(s + m) * d], &chi[s * d..(s + m) * d]);
            let dbs = search::bound_chunk(&f.prog, &ch, &f.objs, false, None);
            for (j, db) in dbs.iter().enumerate() {
                let (l, h) = (
                    &clo[(s + j) * d..(s + j + 1) * d],
                    &chi[(s + j) * d..(s + j + 1) * d],
                );
                for (i, o) in f.objs.iter().enumerate() {
                    let (mn, _) = sampled_range(&f.prog, l, h, o, 17 + j as u64);
                    assert!(
                        db.lbs[i] <= mn + 1e-9 * (1.0 + mn.abs()),
                        "CONTAINMENT: certified lb {} > sampled {mn}",
                        db.lbs[i]
                    );
                    checked += 1;
                }
            }
        }
        for (b, (fate, _)) in fates.iter().enumerate() {
            if matches!(fate, Fate::Closed) {
                let (l, h) = (&lo[b * d..(b + 1) * d], &hi[b * d..(b + 1) * d]);
                let mut r = Lcg(b as u64);
                for _ in 0..500 {
                    let x = sample(&mut r, l, h);
                    let violates_all = f
                        .objs
                        .iter()
                        .zip(&f.thrs)
                        .all(|(o, t)| point_objective(&f.prog, &x, o) <= *t);
                    assert!(!violates_all, "SOUNDNESS: a closed box holds a violation");
                }
            }
        }
    }
    assert!(checked > 10, "VOID: {checked} bounds checked");
}

/// THE SAT MOAT: whatever the backend returns — honest on the probe or not —
/// a threshold a sampled point violates is never verified.
#[test]
fn lowdim_bbe_garbage_backends_never_verify_a_planted_violation() {
    let d = 3;
    let g = relu_net(31, d, 14);
    let (lo, hi) = root_box(d);
    let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
    let mut planted = Vec::new();
    for (o, frac) in [
        (vec![1.0, 0.0], 0.02),
        (vec![0.0, -1.0], 0.3),
        (vec![1.0, -1.0], 0.01),
    ] {
        let (mn, mx) = sampled_range(&prog, &lo, &hi, &o, 55);
        planted.push(Fx {
            prog: prog.clone(),
            lo: lo.clone(),
            hi: hi.clone(),
            objs: vec![o],
            thrs: vec![mn + frac * (mx - mn)],
            cfg: LowDimConfig {
                batch: 32,
                chunk: 4,
                max_queue: 50_000,
                ..LowDimConfig::default()
            },
        });
    }
    let mut searched_claims = 0u64;
    for f in &planted {
        for mode in [
            Mode::ClaimAll,
            Mode::NeverClaim,
            Mode::Nan,
            Mode::Random,
            Mode::FlipGrads,
            Mode::Permute,
            Mode::Anti,
            Mode::Lying,
        ] {
            for honest in [0usize, 1] {
                let b = TestBackend::single(f, mode, honest);
                let (o, _) = with_backend(&b, || run_core(f, &bbe_cfg(&f.cfg), 3));
                assert!(
                    !verified(&o),
                    "SOUNDNESS: {mode:?} (honest probe: {honest}) verified a planted violation ({o:?})"
                );
                let line = take_last_bbe_summary().expect("line");
                searched_claims += field_u64(&line, "claimed");
            }
        }
    }
    assert!(
        searched_claims > 0,
        "VOID: no garbage claim reached the certifier"
    );
}

#[test]
fn lowdim_bbe_claims_are_not_closes() {
    let fx = core_fixtures();
    let f = &fx[2];
    // F3's threshold is violated: no leaf containing the violation closes.
    let cfg = LowDimConfig {
        clip: false,
        ..bbe_cfg(&f.cfg)
    };
    let b = TestBackend::single(f, Mode::ClaimAll, 1);
    let (o, _) = with_backend(&b, || run_core(f, &cfg, 30));
    assert!(!verified(&o), "claims must not close: {o:?}");
    let s = stats_of(&o);
    assert!(s.bbe_claimed > 0, "VOID: nothing claimed");
    assert!(s.bbe_refuted > 0, "a violated root claim must be refuted");
    assert!(s.bbe_certified <= s.closed);
    let line = take_last_bbe_summary().expect("line");
    assert!(field_u64(&line, "refuted") > 0, "{line}");
}

#[test]
fn lowdim_bbe_unclaimed_split_is_an_exact_cover() {
    let fx = core_fixtures();
    let f = &fx[0];
    let d = f.prog.d;
    let cfg = bbe_cfg(&f.cfg);
    let w0: Vec<f64> = f.lo.iter().zip(&f.hi).map(|(l, h)| h - l).collect();
    let wave = Wave {
        prog: &f.prog,
        objs: &f.objs,
        thrs: &f.thrs,
        cfg: &cfg,
        w0: &w0,
        chunk: cfg.chunk,
    };
    let mut rng = Lcg(77);
    let count = 24;
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    for _ in 0..count {
        let a = sample(&mut rng, &f.lo, &f.hi);
        for k in 0..d {
            let r = 0.05 * (rng.next() + 1.1);
            lo.push(a[k] - r);
            hi.push(a[k] + r);
        }
    }
    let b = TestBackend::single(f, Mode::NeverClaim, 1);
    let fates = with_backend(&b, || {
        let mut run = bbe::arm_single(&f.prog, &f.lo, &f.hi, &f.objs, &cfg);
        assert!(run.live());
        match bbe::single_wave(
            &mut run,
            &wave,
            &lo,
            &hi,
            count,
            Instant::now() + Duration::from_secs(60),
        ) {
            bbe::WaveResult::Done(fates) => fates,
            _ => panic!("wave"),
        }
    });
    let mut splits = 0;
    for (b, (fate, note)) in fates.iter().enumerate() {
        assert_eq!(note.claim, ClaimMark::Unclaimed);
        let (l, h) = (&lo[b * d..(b + 1) * d], &hi[b * d..(b + 1) * d]);
        match fate {
            Fate::Split {
                lo: sl,
                hi: sh,
                dim,
                mid,
                shrunk,
            } => {
                assert!(!shrunk, "an unclaimed leaf is never clipped");
                assert_eq!(sl.as_slice(), l, "the split box is the leaf's own");
                assert_eq!(sh.as_slice(), h);
                assert!(*mid > l[*dim] && *mid < h[*dim], "mid strictly inside");
                splits += 1;
            }
            Fate::Candidate(_) | Fate::Resolution => {}
            other => panic!("an unclaimed leaf must never close or requeue: {other:?}"),
        }
    }
    assert!(splits > 0, "VOID: no split");
}

#[test]
fn lowdim_bbe_batches_more_than_one_leaf_and_certifies_closes() {
    let fx = core_fixtures();
    for f in fx.iter().take(2) {
        let b = TestBackend::single(f, Mode::Reference, 0);
        let (o, _) = with_backend(&b, || run_core(f, &bbe_cfg(&f.cfg), 60));
        assert!(verified(&o), "{o:?}");
        let line = take_last_bbe_summary().expect("line");
        assert!(line.starts_with("[bbe] phase=input_split lane=lowdim scheduler=single "));
        assert!(line.ends_with("m8-input-split-batch/v1"), "{line}");
        assert_eq!(field(&line, "fallback"), "none", "{line}");
        assert_eq!(field(&line, "probe"), "ok", "{line}");
        assert!(
            field_u64(&line, "calls") >= 2,
            "probe + at least one wave: {line}"
        );
        assert!(field_u64(&line, "batched_max") > 1, "{line}");
        assert!(field_u64(&line, "certified") > 0, "{line}");
        let s = stats_of(&o);
        assert!(s.bbe_certified > 0 && s.bbe_claimed >= s.bbe_certified);
        assert!(
            (s.closed_volume - 1.0).abs() < 1e-9,
            "a verified clause closes its whole box: vol {}",
            s.closed_volume
        );
    }
}

fn pooled<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool")
        .install(f)
}

#[test]
fn lowdim_bbe_is_thread_count_independent() {
    let fx = core_fixtures();
    for f in fx.iter().take(2) {
        let runs: Vec<_> = [1usize, 3, 8]
            .iter()
            .map(|&t| {
                pooled(t, || {
                    let b = TestBackend::single(f, Mode::Reference, 0);
                    let (o, _) = with_backend(&b, || run_core(f, &bbe_cfg(&f.cfg), 60));
                    let s = stats_of(&o);
                    (
                        counters(&o),
                        s.bbe_claimed,
                        s.bbe_certified,
                        s.bbe_refuted,
                        s.closed_volume.to_bits(),
                    )
                })
            })
            .collect();
        assert_eq!(runs[0], runs[1], "1 vs 3 threads");
        assert_eq!(runs[0], runs[2], "1 vs 8 threads");
    }
}

#[test]
fn lowdim_bbe_reference_backend_reproduces_the_legacy_verdicts() {
    let mut fixtures = core_fixtures();
    fixtures.truncate(2);
    // Add false-margin variants of the first two fixtures.
    for i in 0..2 {
        let f = &fixtures[i];
        let (mn, mx) = sampled_range(&f.prog, &f.lo, &f.hi, &f.objs[0], 91);
        let mut thrs = f.thrs.clone();
        thrs[0] = mn + 0.02 * (mx - mn);
        fixtures.push(Fx {
            prog: f.prog.clone(),
            lo: f.lo.clone(),
            hi: f.hi.clone(),
            objs: f.objs.clone(),
            thrs,
            cfg: f.cfg.clone(),
        });
    }
    let mut compared = 0;
    for f in &fixtures {
        let (legacy, _) = run_core(f, &f.cfg, 60);
        let b = TestBackend::single(f, Mode::Reference, 0);
        let (searched, _) = with_backend(&b, || run_core(f, &bbe_cfg(&f.cfg), 60));
        assert_eq!(
            verified(&legacy),
            verified(&searched),
            "verdict parity: legacy {legacy:?} vs searched {searched:?}"
        );
        if verified(&legacy) {
            let (l, s) = (stats_of(&legacy), stats_of(&searched));
            // f64 evaluations: legacy bounds every domain; the searched lane
            // only its claims (the probe's one root bound aside).
            assert!(
                (s.bbe_claimed as f64) <= 0.6 * l.domains as f64,
                "f64 evaluations {} vs legacy {}",
                s.bbe_claimed,
                l.domains
            );
            compared += 1;
        }
    }
    assert!(compared >= 2, "VOID: {compared} verified trees compared");
}

/// G1: the search copy is the lane program — an evaluator of the
/// `FlProblem` (written here) agrees with the lane's `point_values`.
#[test]
fn lowdim_bbe_search_copy_evaluates_like_the_lane_program() {
    let mut seen_ops = std::collections::BTreeSet::new();
    for (g, d) in [(relu_net(41, 3, 9), 3usize), (lz_net(42, 4, 11), 4)] {
        let (lo, hi) = root_box(d);
        let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
        let problem = bbe::fl_problem(&[vec![&prog]], vec![vec![1.0; prog.sizes[prog.seed]]]);
        problem.validate().expect("the search copy validates");
        let fam = &problem.families[0];
        let mut rng = Lcg(5);
        for _ in 0..50 {
            let x = sample(&mut rng, &lo, &hi);
            let mut vals: Vec<Vec<f64>> = Vec::new();
            for (k, op) in fam.program.ops.iter().enumerate() {
                seen_ops.insert(op.tag());
                let n = fam.program.sizes[k];
                let v: Vec<f64> = match (op, &fam.clauses[0][k]) {
                    (FlOp::Input, _) => x.clone(),
                    (FlOp::Linear(l), FlOpConsts::Bias { lo, hi }) => (0..n)
                        .map(|i| {
                            f64::midpoint(lo[i], hi[i])
                                + (0..l.n_in as usize)
                                    .map(|j| l.w[i * l.n_in as usize + j] * vals[l.src as usize][j])
                                    .sum::<f64>()
                        })
                        .collect(),
                    (FlOp::Relu { src }, _) => {
                        vals[*src as usize].iter().map(|v| v.max(0.0)).collect()
                    }
                    (FlOp::Slice { src, start }, _) => {
                        vals[*src as usize][*start as usize..*start as usize + n].to_vec()
                    }
                    (FlOp::Concat { srcs }, _) => srcs
                        .iter()
                        .flat_map(|&s| vals[s as usize].clone())
                        .collect(),
                    (FlOp::Identity { src }, _) => vals[*src as usize].clone(),
                    (FlOp::Sparse { terms }, FlOpConsts::Sparse { coef, k_lo, k_hi }) => {
                        let mut at = 0;
                        (0..n)
                            .map(|j| {
                                let mut acc = f64::midpoint(k_lo[j], k_hi[j]);
                                for &(s, i) in &terms[j] {
                                    acc += coef[at] * vals[s as usize][i as usize];
                                    at += 1;
                                }
                                acc
                            })
                            .collect()
                    }
                    other => panic!("unexpected op in the fixture: {other:?}"),
                };
                vals.push(v);
            }
            let ours = &vals[fam.program.seed];
            let lane = point_values(&prog, &x);
            assert_eq!(ours.len(), lane.len());
            for (a, b) in ours.iter().zip(&lane) {
                assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()), "{a} vs {b}");
            }
        }
    }
    for tag in ["Input", "Linear", "Relu", "Slice", "Concat"] {
        assert!(
            seen_ops.contains(tag),
            "VOID: no {tag} op exercised ({seen_ops:?})"
        );
    }
}

#[test]
fn lowdim_bbe_refutation_storm_latches_legacy() {
    let fx = core_fixtures();
    // A FLAT net: every ReLU is inactive on the whole box, so the output is
    // the constant last bias `c`. With the threshold `c + 1e-7` (below the
    // candidate margin) no leaf ever closes and no centre is a candidate: the
    // tree only grows, breadth first (batch above the frontier), until the
    // queue cap. The anti backend claims every leaf (none closes), so the
    // refuted share is 1 and the storm latch must fire after 1024 claims.
    let flat = {
        let d = 2;
        let mut g = GraphNetwork::new();
        let w1 = Array2::from_shape_fn((3, d), |(i, j)| 0.01 * (i + j) as f32);
        let b1 = Array1::from_elem(3, -5.0f32);
        g.add_node(GraphNode::new(
            "l1",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("linear")),
            vec![NETWORK_INPUT.to_string()],
        ));
        g.add_node(GraphNode::new(
            "r1",
            Layer::ReLU(ReLULayer::new()),
            vec!["l1".into()],
        ));
        let w2 = Array2::from_shape_fn((2, 3), |(i, j)| 0.5 - 0.25 * (i + j) as f32);
        let b2 = Array1::from_vec(vec![0.3f32, 0.1]);
        g.add_node(GraphNode::new(
            "l2",
            Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("linear")),
            vec!["r1".into()],
        ));
        g.set_output("l2");
        let (lo, hi) = root_box(d);
        let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
        Fx {
            prog,
            lo,
            hi,
            objs: vec![vec![1.0, 0.0]],
            thrs: vec![f64::from(0.3f32) + 1e-7],
            cfg: LowDimConfig {
                batch: 4096,
                chunk: 8,
                max_queue: 200_000,
                ..LowDimConfig::default()
            },
        }
    };
    let f = &flat;
    let legacy = run_core(f, &f.cfg, 60);
    assert!(
        counters(&legacy.0).2 >= 2048,
        "VOID precondition: the storm fixture needs a large tree ({:?})",
        counters(&legacy.0)
    );
    assert!(!fx.is_empty());
    // Honest on the probe, then claims exactly what will NOT close: every
    // claim refuted, every closable leaf split unbounded — until the storm
    // latch hands the run back to the legacy bounder.
    let cfg = bbe_cfg(&f.cfg);
    let b = TestBackend::single(f, Mode::Anti, 1);
    let (o, _) = with_backend(&b, || run_core(f, &cfg, 60));
    let line = take_last_bbe_summary().expect("line");
    assert_eq!(field(&line, "fallback"), "refutation_storm", "{line}");
    assert!(field_u64(&line, "claimed") >= 1024, "{line}");
    assert_eq!(field_u64(&line, "certified"), 0, "{line}");
    assert_eq!(
        verified(&o),
        verified(&legacy.0),
        "the latched run finishes on legacy"
    );
    // The latch itself.
    let b = TestBackend::single(f, Mode::Reference, 0);
    with_backend(&b, || {
        let mut run = bbe::arm_single(&f.prog, &f.lo, &f.hi, &f.objs, &bbe_cfg(&f.cfg));
        assert!(run.live());
        run.note(1000, 600, 400, 0);
        assert!(run.live(), "below the claim floor");
        run.note(100, 0, 100, 0);
        assert!(run.live(), "refuted 500 of 1100: not a storm");
        run.note(100, 0, 100, 0);
        assert!(
            run.live(),
            "refuted 600 of 1200: exactly half is not a storm"
        );
        run.note(10, 0, 10, 0);
        assert!(!run.live(), "refuted 610 of 1210: a storm");
        assert_eq!(run.stats.fallback, Some("refutation_storm"));
    });
}

/// The ingest-gate SOURCE RATCHET: `lowdim` has exactly one `into_ingest`
/// call (in `bbe.rs`), and neither `bbe.rs` nor `disjunction.rs` constructs a
/// `DomainBound` (a hint never becomes a bound).
#[test]
fn lowdim_bbe_has_one_ingest_gate_and_hints_never_build_a_domain_bound() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = vec![root.join("lowdim.rs")];
    for e in std::fs::read_dir(root.join("lowdim")).expect("lowdim dir") {
        let p = e.expect("entry").path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let is_rs = p.extension().is_some_and(|ext| ext == "rs");
        if is_rs && !name.ends_with("_tests.rs") && name != "tests.rs" {
            files.push(p);
        }
    }
    assert!(files.len() >= 10, "VOID: {} files", files.len());
    let code = |text: &str| -> String {
        text.lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut ingest_sites = Vec::new();
    for p in &files {
        let text = std::fs::read_to_string(p).expect("readable");
        let n = code(&text).matches("into_ingest(").count();
        for _ in 0..n {
            ingest_sites.push(p.file_name().unwrap().to_string_lossy().to_string());
        }
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name == "bbe.rs" || name == "disjunction.rs" {
            assert!(
                !code(&text).contains("DomainBound {"),
                "{name} constructs a DomainBound: a hint must never become a bound"
            );
        }
    }
    assert_eq!(
        ingest_sites,
        vec!["bbe.rs".to_string()],
        "the ONE ingest gate"
    );
}

// =====================================================================
//  Shared worklist (fold) under the wave search
// =====================================================================

fn shared_backend(fams: &[Vec<FamClause>], mode: Mode, honest: usize) -> Arc<TestBackend> {
    let lane: Vec<Vec<Program>> = fams
        .iter()
        .map(|f| f.iter().map(|c| c.prog.clone()).collect())
        .collect();
    let thr: Vec<f64> = fams.iter().flatten().flat_map(|c| c.thrs.clone()).collect();
    TestBackend::new(lane, mode, honest).with_thresholds(thr)
}

fn shared_run(
    fams: Vec<Vec<FamClause>>,
    cfg: &LowDimConfig,
    secs: u64,
) -> (Vec<Vec<LowDimOutcome>>, shared::SharedStats) {
    shared::prove_families_shared(fams, Instant::now() + Duration::from_secs(secs), cfg)
}

#[test]
fn shared_worklist_bbe_certified_bounds_never_exceed_a_sampled_value() {
    let (fams, _) = family_fixtures();
    let cfg = bbe_cfg(&fam_cfg());
    let b = shared_backend(&fams, Mode::Reference, 0);
    let live: Vec<Vec<bool>> = fams.iter().map(|f| vec![true; f.len()]).collect();
    let mut checked = 0usize;
    with_backend(&b, || {
        let (mut run, obj_base) = bbe::arm_shared(&fams, &live, &cfg);
        assert!(run.live(), "reference passes the probe");
        let mut rng = Lcg(3);
        let mut groups = Vec::new();
        for (f, fam) in fams.iter().enumerate() {
            let mut which = Vec::new();
            let (mut lo, mut hi) = (Vec::new(), Vec::new());
            for _ in 0..12 {
                let c = (f64::midpoint(rng.next(), 1.0) * fam.len() as f64) as usize % fam.len();
                let cl = &fam[c];
                let a = sample(&mut rng, &cl.lo, &cl.hi);
                for k in 0..a.len() {
                    let r = 0.05 + 0.1 * (rng.next() + 1.0);
                    lo.push((a[k] - r).max(cl.lo[k]));
                    hi.push((a[k] + r).min(cl.hi[k]));
                }
                which.push(c);
            }
            groups.push(bbe::GroupIn { f, which, lo, hi });
        }
        let dims: Vec<usize> = fams.iter().map(|f| f[0].prog.d).collect();
        let chunks = vec![cfg.chunk; fams.len()];
        let w0s: Vec<Vec<Vec<f64>>> = fams
            .iter()
            .map(|f| {
                f.iter()
                    .map(|c| c.lo.iter().zip(&c.hi).map(|(l, h)| h - l).collect())
                    .collect()
            })
            .collect();
        let per = match bbe::group_wave(
            &mut run,
            &fams,
            &obj_base,
            &groups,
            &dims,
            &chunks,
            &w0s,
            &cfg,
            Instant::now() + Duration::from_secs(60),
        ) {
            bbe::WaveResult::Done(per) => per,
            _ => panic!("group wave"),
        };
        for (g, fates) in groups.iter().zip(&per) {
            let d = dims[g.f];
            let fam = &fams[g.f];
            let claimed: Vec<usize> = (0..g.which.len())
                .filter(|&b| fates[b].1 != ClaimMark::Unclaimed)
                .collect();
            let progs: Vec<&Program> = fam.iter().map(|c| &c.prog).collect();
            let which: Vec<usize> = claimed.iter().map(|&b| g.which[b]).collect();
            let lo: Vec<f64> = claimed
                .iter()
                .flat_map(|&b| g.lo[b * d..(b + 1) * d].to_vec())
                .collect();
            let hi: Vec<f64> = claimed
                .iter()
                .flat_map(|&b| g.hi[b * d..(b + 1) * d].to_vec())
                .collect();
            let objs: Vec<&[Vec<f64>]> = which.iter().map(|&c| fam[c].objs.as_slice()).collect();
            let bounds = shared::bound_group(&progs, &which, &lo, &hi, &objs, cfg.chunk, false);
            for (j, db) in bounds.iter().enumerate() {
                let c = which[j];
                let (l, h) = (&lo[j * d..(j + 1) * d], &hi[j * d..(j + 1) * d]);
                let (mn, _) = sampled_range(&fam[c].prog, l, h, &fam[c].objs[0], j as u64);
                assert!(
                    db.lbs[0] <= mn + 1e-9 * (1.0 + mn.abs()),
                    "CONTAINMENT: shared certified lb {} > sampled {mn}",
                    db.lbs[0]
                );
                checked += 1;
            }
        }
    });
    assert!(checked > 0, "VOID: nothing claimed");
}

#[test]
fn shared_worklist_bbe_batches_more_than_one_leaf_and_certifies_closes() {
    let (fams, truth) = family_fixtures();
    let legacy = shared_run(family_fixtures().0, &fam_cfg(), 60).0;
    let b = shared_backend(&fams, Mode::Reference, 0);
    let (outs, st) = with_backend(&b, || shared_run(fams, &bbe_cfg(&fam_cfg()), 60));
    let line = take_last_bbe_summary().expect("line");
    assert!(
        line.starts_with("[bbe] phase=input_split lane=lowdim scheduler=shared "),
        "{line}"
    );
    assert_eq!(field(&line, "fallback"), "none", "{line}");
    assert!(field_u64(&line, "calls") > 2, "{line}");
    assert!(field_u64(&line, "batched_max") > 1, "{line}");
    assert!(field_u64(&line, "certified") > 0, "{line}");
    assert!(st.vol > 0.0);
    for (f, fam) in outs.iter().enumerate() {
        for (c, o) in fam.iter().enumerate() {
            assert_eq!(
                verified(o),
                verified(&legacy[f][c]),
                "family {f} clause {c}"
            );
            if !truth[f][c] {
                assert!(!verified(o), "SOUNDNESS: family {f} clause {c}");
            }
        }
    }
}

#[test]
fn shared_worklist_bbe_never_verifies_a_false_bound() {
    let mut claims = 0u64;
    for mode in [
        Mode::ClaimAll,
        Mode::NeverClaim,
        Mode::Nan,
        Mode::Random,
        Mode::FlipGrads,
        Mode::Permute,
        Mode::Anti,
    ] {
        let (fams, truth) = family_fixtures();
        let honest = fams.len();
        let b = shared_backend(&fams, mode, honest);
        let cfg = LowDimConfig {
            max_queue: 50_000,
            ..bbe_cfg(&fam_cfg())
        };
        let (outs, _) = with_backend(&b, || shared_run(fams, &cfg, 3));
        for (f, fam) in outs.iter().enumerate() {
            for (c, o) in fam.iter().enumerate() {
                if !truth[f][c] {
                    assert!(
                        !verified(o),
                        "SOUNDNESS: {mode:?} verified family {f} clause {c}'s violated bound"
                    );
                }
            }
        }
        claims += field_u64(&take_last_bbe_summary().expect("line"), "claimed");
    }
    assert!(claims > 0, "VOID: no garbage claim reached the certifier");
}

// =====================================================================
//  The disjunction tree
// =====================================================================

/// A same-box disjunction over `relu_net(seed)`: clauses with true margins
/// and (per `violated`) planted violations; returns the admission, the
/// clauses and each clause's truth.
fn disjunction_fixture(
    seed: u64,
    d: usize,
    violated: &[bool],
) -> (
    GraphNetwork,
    LaneAdmission,
    Vec<DisjunctionClause>,
    Vec<bool>,
) {
    let g = relu_net(seed, d, 12);
    let (lo, hi) = root_box(d);
    let admission = admit(&g, &boxed(&lo, &hi)).expect("admits");
    let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
    let objs = [
        vec![1.0f32, 0.0],
        vec![0.0, 1.0],
        vec![1.0, -1.0],
        vec![-1.0, 0.5],
    ];
    let mut clauses = Vec::new();
    let mut truth = Vec::new();
    for (c, &bad) in violated.iter().enumerate() {
        let o32 = objs[c % objs.len()].clone();
        let o: Vec<f64> = o32.iter().map(|&v| f64::from(v)).collect();
        let (mn, mx) = sampled_range(&prog, &lo, &hi, &o, 70 + c as u64);
        let thr = if bad {
            mn + 0.04 * (mx - mn)
        } else {
            mn - 0.1 * (mx - mn)
        };
        // A second conjunct that never closes (the conjunction closes on the
        // first alone) for odd clauses.
        let (mut os, mut ts) = (vec![o32], vec![thr as f32]);
        if c % 2 == 1 {
            let o2 = objs[(c + 1) % objs.len()].clone();
            let o2f: Vec<f64> = o2.iter().map(|&v| f64::from(v)).collect();
            let (_, mx2) = sampled_range(&prog, &lo, &hi, &o2f, 80 + c as u64);
            os.push(o2);
            ts.push((mx2 + 1.0) as f32);
        }
        clauses.push(DisjunctionClause {
            objectives: os,
            thresholds: ts,
        });
        // f32 rounding of the threshold keeps a 4%/10%-of-range margin.
        truth.push(!bad);
    }
    (g, admission, clauses, truth)
}

fn dis_cfg() -> LowDimConfig {
    LowDimConfig {
        batch: 32,
        chunk: 4,
        max_queue: 100_000,
        engine: InputSplitEngine::Bbe,
        ..LowDimConfig::default()
    }
}

fn dis_backend(
    admission: &LaneAdmission,
    clauses: &[DisjunctionClause],
    mode: Mode,
    honest: usize,
) -> Arc<TestBackend> {
    let thr: Vec<f64> = clauses
        .iter()
        .flat_map(|c| c.thresholds.iter().map(|&t| f64::from(t)))
        .collect();
    TestBackend::new(vec![vec![admission.prog.clone()]], mode, honest).with_thresholds(thr)
}

#[test]
fn disjunction_agrees_with_one_tree_per_clause() {
    for (seed, violated) in [
        (51u64, vec![false, false, false]),
        (52, vec![false, true, false, false]),
    ] {
        let (_, admission, clauses, truth) = disjunction_fixture(seed, 3, &violated);
        assert!(disjunction_admits(clauses.len(), &admission).is_ok());
        let cfg = dis_cfg();
        let deadline = Instant::now() + Duration::from_secs(60);
        let one = prove_disjunction_admitted(&admission, &clauses, deadline, &cfg);
        let mut sum_domains = 0u64;
        for (c, cl) in clauses.iter().enumerate() {
            let alone = prove_conjunction_admitted(
                &admission,
                &cl.objectives,
                &cl.thresholds,
                deadline,
                &cfg,
                &mut None,
            );
            sum_domains += counters(&alone).2;
            assert_eq!(
                verified(&one.per_clause[c]),
                verified(&alone),
                "seed {seed} clause {c}: disjunction {:?} vs alone {alone:?}",
                one.per_clause[c]
            );
            assert_eq!(verified(&alone), truth[c], "fixture truth, clause {c}");
        }
        assert!(
            one.stats.domains <= sum_domains,
            "seed {seed}: one tree {} domains vs {} over per-clause trees",
            one.stats.domains,
            sum_domains
        );
    }
}

#[test]
fn disjunction_drops_a_clause_only_on_its_own_strict_close() {
    let (_, admission, _, _) = disjunction_fixture(53, 2, &[false, false]);
    let prog = admission.prog;
    // Three single-conjunct clauses over rows 0, 1, 2; clause 1 has two rows.
    let objs = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let thrs = vec![0.5, 1.0, 2.0, -3.0];
    let rows_of = vec![0..1, 1..3, 3..4];
    let cfg = LowDimConfig::default();
    let w0 = vec![1.0, 1.0];
    let ctx = disjunction::test_ctx(&prog, &objs, &thrs, &rows_of, &cfg, &w0);
    let db = |lbs: Vec<f64>| search::DomainBound {
        lbs,
        grads: vec![0.0; 8],
        planes: Vec::new(),
        joint_hits: 0,
        alpha: search::AlphaNote::default(),
    };
    let u = vec![0usize, 1, 2, 3];
    let (l, h) = (vec![-0.1, -0.1], vec![0.1, 0.1]);
    // Row 0 at its threshold (NOT strict): clause 0 stays; row 2 above its
    // threshold: clause 1 clears (any conjunct); row 3 below: clause 2 stays.
    let dec =
        disjunction::test_decide(&ctx, &l, &h, &[0, 1, 2], &u, &db(vec![0.5, 0.0, 2.5, -4.0]));
    assert_eq!(dec, vec![1], "only clause 1 closes, strictly");
    let dec = disjunction::test_decide(
        &ctx,
        &l,
        &h,
        &[0, 1, 2],
        &u,
        &db(vec![f64::from_bits(0.5f64.to_bits() + 1), 1.0, 2.0, -3.0]),
    );
    assert_eq!(
        dec,
        vec![0],
        "clause 0 one ulp above; rows at thresholds stay open"
    );
    let dec = disjunction::test_decide(&ctx, &l, &h, &[0, 2], &u, &db(vec![1.0, 9.0, 9.0, 0.0]));
    assert_eq!(
        dec,
        vec![0, 2],
        "clause 1 is not in the mask: never cleared"
    );
}

#[test]
fn disjunction_never_verifies_a_clause_with_a_sampled_violation() {
    let mut claims = 0u64;
    for (seed, violated) in [
        (54u64, vec![true, false, true]),
        (55, vec![false, true, true, false]),
    ] {
        let (_, admission, clauses, truth) = disjunction_fixture(seed, 3, &violated);
        // Legacy bounder, then the wave search with every garbage backend.
        let mut runs = vec![prove_disjunction_admitted(
            &admission,
            &clauses,
            Instant::now() + Duration::from_secs(10),
            &dis_cfg(),
        )];
        for mode in [
            Mode::Reference,
            Mode::ClaimAll,
            Mode::NeverClaim,
            Mode::Nan,
            Mode::Random,
            Mode::FlipGrads,
            Mode::Anti,
        ] {
            let b = dis_backend(&admission, &clauses, mode, 1);
            runs.push(with_backend(&b, || {
                prove_disjunction_admitted(
                    &admission,
                    &clauses,
                    Instant::now() + Duration::from_secs(3),
                    &bbe_cfg(&dis_cfg()),
                )
            }));
            claims += field_u64(&take_last_bbe_summary().expect("line"), "claimed");
        }
        for run in &runs {
            for (c, o) in run.per_clause.iter().enumerate() {
                if !truth[c] {
                    assert!(
                        !verified(o),
                        "SOUNDNESS: seed {seed} clause {c} verified ({o:?})"
                    );
                }
            }
        }
    }
    assert!(claims > 0, "VOID: no claim reached the certifier");
}

#[test]
fn disjunction_clip_hull_keeps_every_open_clause_violation() {
    let (_, admission, clauses, _) = disjunction_fixture(56, 2, &[true, false, true]);
    let prog = admission.prog;
    let objs: Vec<Vec<f64>> = clauses
        .iter()
        .flat_map(|c| {
            c.objectives
                .iter()
                .map(|o| o.iter().map(|&v| f64::from(v)).collect())
        })
        .collect();
    let thrs: Vec<f64> = clauses
        .iter()
        .flat_map(|c| c.thresholds.iter().map(|&t| f64::from(t)))
        .collect();
    let mut rows_of = Vec::new();
    let mut at = 0;
    for c in &clauses {
        rows_of.push(at..at + c.objectives.len());
        at += c.objectives.len();
    }
    let cfg = LowDimConfig {
        clip: true,
        ..LowDimConfig::default()
    };
    let (lo0, hi0) = root_box(2);
    let w0: Vec<f64> = lo0.iter().zip(&hi0).map(|(l, h)| h - l).collect();
    let ctx = disjunction::test_ctx(&prog, &objs, &thrs, &rows_of, &cfg, &w0);
    let mut rng = Lcg(9);
    let mut hulls = 0;
    let mut checked = 0;
    for _ in 0..30 {
        let a = sample(&mut rng, &lo0, &hi0);
        let r = 0.1 + 0.2 * (rng.next() + 1.0);
        let l: Vec<f64> = a.iter().zip(&lo0).map(|(v, lo)| (v - r).max(*lo)).collect();
        let h: Vec<f64> = a.iter().zip(&hi0).map(|(v, hi)| (v + r).min(*hi)).collect();
        let mask: Vec<usize> = (0..clauses.len()).collect();
        let (cleared, remaining, hull) = disjunction::test_decide_full(&ctx, &l, &h, &mask);
        if let Some((hl, hh)) = &hull {
            if hl != &l || hh != &h {
                hulls += 1;
            }
        }
        let mut pr = Lcg(checked as u64);
        for _ in 0..400 {
            let x = sample(&mut pr, &l, &h);
            for (c, rows) in rows_of.iter().enumerate() {
                let violates = rows
                    .clone()
                    .all(|q| point_objective(&prog, &x, &objs[q]) <= thrs[q]);
                if !violates {
                    continue;
                }
                assert!(
                    !cleared.contains(&c),
                    "SOUNDNESS: clause {c} cleared on a box holding its violation"
                );
                if remaining.contains(&c) {
                    let (hl, hh) = hull.as_ref().expect("an open clause keeps a box");
                    assert!(
                        (0..x.len()).all(|k| x[k] >= hl[k] && x[k] <= hh[k]),
                        "SOUNDNESS: clause {c}'s violation left the hull"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "VOID: no violating sample");
    assert!(hulls > 0, "VOID: the clip never shrank a box");
}

#[test]
fn disjunction_candidate_gives_up_only_its_clause() {
    let (_, admission, mut clauses, _) = disjunction_fixture(57, 3, &[false, false]);
    // Clause 0: violated at every point (a threshold above the maximum).
    let prog = admission.prog.clone();
    let (lo, hi) = root_box(3);
    let o: Vec<f64> = clauses[0].objectives[0]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let (_, mx) = sampled_range(&prog, &lo, &hi, &o, 1);
    clauses[0].thresholds[0] = (mx + 1.0) as f32;
    let mut seen = Vec::new();
    let out = prove_disjunction_admitted_with(
        &admission,
        &clauses,
        Instant::now() + Duration::from_secs(60),
        &dis_cfg(),
        &mut |c: usize, x: &[f64]| {
            seen.push((c, x.to_vec()));
            false
        },
    );
    match &out.per_clause[0] {
        LowDimOutcome::Unproved(_, stop) => assert_eq!(*stop, LowDimStop::Candidate),
        other => panic!("clause 0 must stop on a candidate: {other:?}"),
    }
    assert!(out.witnesses[0].is_some() && out.witnesses[1].is_none());
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].0, 0);
    assert!(
        verified(&out.per_clause[1]),
        "clause 1 still verifies: {:?}",
        out.per_clause[1]
    );
    // A confirmed candidate stops the run.
    let out = prove_disjunction_admitted_with(
        &admission,
        &clauses,
        Instant::now() + Duration::from_secs(60),
        &dis_cfg(),
        &mut |_: usize, _: &[f64]| true,
    );
    assert!(matches!(
        out.per_clause[0],
        LowDimOutcome::Unproved(_, LowDimStop::Candidate)
    ));
    assert!(
        matches!(
            out.per_clause[1],
            LowDimOutcome::Verified(_) | LowDimOutcome::Unproved(_, LowDimStop::Candidate)
        ),
        "a confirmed candidate stops the run: {:?}",
        out.per_clause[1]
    );
}

#[test]
fn disjunction_is_thread_count_independent() {
    let (_, admission, clauses, _) = disjunction_fixture(58, 3, &[false, true, false]);
    let key = |o: &DisjunctionOutcome| {
        (
            o.per_clause.iter().map(counters).collect::<Vec<_>>(),
            o.stats.domains,
            o.stats.waves,
            o.stats.mask_drops,
            o.stats
                .closed_volume
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            o.witnesses
                .iter()
                .map(|w| {
                    w.as_ref()
                        .map(|x| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                })
                .collect::<Vec<_>>(),
        )
    };
    for search in [false, true] {
        let runs: Vec<_> = [1usize, 3, 8]
            .iter()
            .map(|&t| {
                pooled(t, || {
                    let cfg = if search {
                        bbe_cfg(&dis_cfg())
                    } else {
                        dis_cfg()
                    };
                    let b = dis_backend(&admission, &clauses, Mode::Reference, 0);
                    let o = with_backend(&b, || {
                        prove_disjunction_admitted(
                            &admission,
                            &clauses,
                            Instant::now() + Duration::from_secs(60),
                            &cfg,
                        )
                    });
                    key(&o)
                })
            })
            .collect();
        assert_eq!(runs[0], runs[1], "search={search}: 1 vs 3 threads");
        assert_eq!(runs[0], runs[2], "search={search}: 1 vs 8 threads");
    }
}

#[test]
fn disjunction_bbe_batches_more_than_one_leaf_and_certifies_closes() {
    let (_, admission, clauses, truth) = disjunction_fixture(59, 3, &[false, false, true]);
    let legacy = prove_disjunction_admitted(
        &admission,
        &clauses,
        Instant::now() + Duration::from_secs(60),
        &dis_cfg(),
    );
    let b = dis_backend(&admission, &clauses, Mode::Reference, 0);
    let out = with_backend(&b, || {
        prove_disjunction_admitted(
            &admission,
            &clauses,
            Instant::now() + Duration::from_secs(60),
            &bbe_cfg(&dis_cfg()),
        )
    });
    let line = take_last_bbe_summary().expect("line");
    assert!(
        line.starts_with("[bbe] phase=input_split lane=lowdim scheduler=disjunction "),
        "{line}"
    );
    assert_eq!(field(&line, "fallback"), "none", "{line}");
    assert!(field_u64(&line, "batched_max") > 1, "{line}");
    assert!(field_u64(&line, "certified") > 0, "{line}");
    assert!(out.stats.bbe.calls >= 2);
    for c in 0..clauses.len() {
        assert_eq!(
            verified(&out.per_clause[c]),
            verified(&legacy.per_clause[c]),
            "clause {c}"
        );
        assert_eq!(verified(&out.per_clause[c]), truth[c], "clause {c}");
    }
}

/// Review fix (an untrusted miss is never a terminal stop): a backend that
/// NEVER claims drives every leaf of a true margin down to the resolution
/// floor. There an unsplittable leaf must be certified by the lane's own f64
/// bound (and counted as a claim), not end the lane (single conjunction) or
/// give its clause up (shared worklist, disjunction tree) on the hint alone.
/// At a floor where the legacy bounder still proves everything, every entry
/// keeps the legacy verdicts, and every close is a certified one. Without the
/// fix each entry stops `Unproved(Resolution)` here.
#[test]
fn lowdim_bbe_resolution_floor_is_certified_never_a_hint_stop() {
    const FLOORS: [f64; 6] = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01];
    let secs = 60;
    // (1) The single conjunction (F1: a true margin).
    let fx = core_fixtures();
    let f = &fx[0];
    let single = |w: f64| LowDimConfig {
        min_width: w,
        max_queue: 200_000,
        ..f.cfg.clone()
    };
    let floor = FLOORS
        .iter()
        .copied()
        .find(|&w| verified(&run_core(f, &single(w), secs).0))
        .expect("VOID precondition: the legacy lane proves F1 at some floor");
    let b = TestBackend::single(f, Mode::NeverClaim, 1);
    let (o, _) = with_backend(&b, || run_core(f, &bbe_cfg(&single(floor)), secs));
    let line = take_last_bbe_summary().expect("line");
    assert_eq!(field(&line, "fallback"), "none", "{line}");
    assert!(
        verified(&o),
        "single: a hint that never claims ended a true margin at floor {floor}: {o:?} ({line})"
    );
    let s = stats_of(&o);
    assert!(
        s.bbe_claimed > 0,
        "VOID: the floor was never reached ({line})"
    );
    assert_eq!(s.closed, s.bbe_certified, "single: {s:?}");
    // (2) The shared worklist (true clauses of the family fixtures).
    let (_, truth) = family_fixtures();
    let fam_floor = |w: f64| LowDimConfig {
        min_width: w,
        max_queue: 200_000,
        ..fam_cfg()
    };
    let proves_true = |outs: &[Vec<LowDimOutcome>]| {
        outs.iter()
            .zip(&truth)
            .all(|(fam, tr)| fam.iter().zip(tr).all(|(o, &t)| !t || verified(o)))
    };
    let floor = FLOORS
        .iter()
        .copied()
        .find(|&w| proves_true(&shared_run(family_fixtures().0, &fam_floor(w), secs).0))
        .expect("VOID precondition: the legacy shared worklist proves the true clauses");
    let (fams, _) = family_fixtures();
    let b = shared_backend(&fams, Mode::NeverClaim, fams.len());
    let (outs, _) = with_backend(&b, || shared_run(fams, &bbe_cfg(&fam_floor(floor)), secs));
    let line = take_last_bbe_summary().expect("line");
    assert_eq!(field(&line, "fallback"), "none", "{line}");
    assert!(
        field_u64(&line, "claimed") > 0,
        "VOID: the floor was never reached ({line})"
    );
    assert!(
        proves_true(&outs),
        "shared: a hint that never claims gave a true clause up at floor {floor}: {outs:?}"
    );
    for (fam, tr) in outs.iter().zip(&truth) {
        for (o, &t) in fam.iter().zip(tr) {
            if t {
                let s = stats_of(o);
                assert_eq!(s.closed, s.bbe_certified, "shared: {s:?}");
            }
        }
    }
    // (3) The disjunction tree (three true clauses on one box).
    let (_, admission, clauses, _) = disjunction_fixture(51, 3, &[false, false, false]);
    let dis_floor = |w: f64| LowDimConfig {
        min_width: w,
        max_queue: 200_000,
        ..dis_cfg()
    };
    let floor = FLOORS
        .iter()
        .copied()
        .find(|&w| {
            prove_disjunction_admitted(
                &admission,
                &clauses,
                Instant::now() + Duration::from_secs(secs),
                &dis_floor(w),
            )
            .per_clause
            .iter()
            .all(verified)
        })
        .expect("VOID precondition: the legacy disjunction tree proves every clause");
    let b = dis_backend(&admission, &clauses, Mode::NeverClaim, 1);
    let out = with_backend(&b, || {
        prove_disjunction_admitted(
            &admission,
            &clauses,
            Instant::now() + Duration::from_secs(secs),
            &bbe_cfg(&dis_floor(floor)),
        )
    });
    let line = take_last_bbe_summary().expect("line");
    assert_eq!(field(&line, "fallback"), "none", "{line}");
    assert!(
        field_u64(&line, "claimed") > 0,
        "VOID: the floor was never reached ({line})"
    );
    for (c, o) in out.per_clause.iter().enumerate() {
        assert!(
            verified(o),
            "disjunction: a hint that never claims gave clause {c} up at floor {floor}: {o:?}"
        );
        // Every leaf that cleared `c` was a certified (claimed) leaf.
        let s = stats_of(o);
        assert!(s.closed <= s.bbe_claimed, "disjunction clause {c}: {s:?}");
    }
}

/// The router rows name functions that exist.
#[test]
fn lowdim_bbe_router_rows_name_real_admissions() {
    let bbe_src = include_str!("bbe.rs");
    let dis_src = include_str!("disjunction.rs");
    assert_eq!(ROUTER_ROWS.len(), 2);
    assert!(bbe_src.contains("pub(crate) fn bbe_admits("));
    assert!(dis_src.contains("pub fn disjunction_admits("));
    assert_eq!(ROUTER_ROWS[0].slot, "lowdim.batch_engine");
    assert_eq!(ROUTER_ROWS[1].slot, "input_split.engine");
}
