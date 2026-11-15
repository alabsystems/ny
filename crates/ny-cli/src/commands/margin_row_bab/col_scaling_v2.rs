// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `col-scaling-v2`: the margin-row pass-cost probe, version 2 (margin-row
//! throughput plan, build B1(b)).
//!
//! # Why v1 (`col-scaling`) is not enough
//!
//! v1 times single Lower passes in ASCENDING R order on the GLOBAL pool and
//! reports WALL only. Two things make its curve unusable as a CPU-cost model:
//!
//! * `net.rs::par_min_len` runs a whole conv as ONE rayon task when
//!   `tensor x r_blk < 16Ki` elements, and at R = 1 `r_blk` is 1, so every
//!   resnet_medium backward conv at R = 1 is serial. The standalone R = 1 wall
//!   is therefore SERIAL WALL, not work — inside the 20-domain concurrent tree a
//!   narrow pass costs roughly its wall in CORE-seconds, not 20x its wall.
//! * Ascending order confounds R with warm-up and allocator state.
//!
//! v2 measures CPU directly: every arm runs on a private 1-thread pool (CPU work
//! with no serialization confound) AND a 20-thread pool, in a SHUFFLED R order,
//! reporting median wall and median process CPU.
//!
//! # Arms
//!
//! * `pass` — one certified Lower pass under deep-domain gates with the scored
//!   eval's collection shape (`unst_abs`), at every R, in Outward AND Parity
//!   mode. `outward/parity` CPU is the certified error lane's cost multiplier;
//!   the plan's B3 (EEL) gate reads it at R >= 64 on the 1-thread pool.
//! * `absw` — every conv op's backward kernel ALONE at r = 1, signed weights
//!   and `|W|` (the error lane's conv), summed over the net.
//! * `stack` — K deep domains x nf columns: K CONCURRENT `run_collect` calls
//!   (what the tree does today) against ONE `run_domain_stacked_collect` pass,
//!   wall and CPU, plus a bit-identity check of the concretized columns. The
//!   plan's B5 gate reads stacked/concurrent CPU at K >= 8, nf <= 3.
//! * `eel` (#mr-errlane, plan B3; opt-in, not in the default arm set) — the
//!   same Outward pass under each certified error lane (`off`, `envelope`,
//!   `per_channel`), CPU per lane at every R, plus the PER-PASS LOOSENESS of
//!   each envelope against the published per-column bound on the same pass
//!   (`max_loosen`, `max_rel` over positive columns, `pos_lost`). `envelope`
//!   engages only at `R >= 16` (`env_engaged=`), exactly as on the lane.
//!
//! # Invocation
//!
//! Arguments ride in the probe selector — this probe deliberately reads NO
//! environment variable (the ny-levers source frontier counts every raw
//! reader site) and needs NO in-tree `benchmarks/`, so a binary built in a
//! worktree runs it against the main tree's corpus by absolute path:
//!
//! ```text
//! ny vnncomp-research margin-row \
//!   'col-scaling-v2,onnx=/abs/model.onnx,vnnlib=/abs/prop.vnnlib[,threads=1+20]
//!    [,cols=1+2+3+4+8+16+64+192][,reps=5][,seed=1][,depth=20][,k=4+8+20]
//!    [,nfk=1+3][,arms=pass+absw+stack[+eel]]'
//! ```
//!
//! (one argument, no spaces). `onnx` and `vnnlib` are required.
//!
//! # CPU clock
//!
//! Process CPU is `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` — the same
//! quantity as `getrusage(RUSAGE_SELF)` user+sys, at ns resolution — via
//! `rustix` (already a unix dependency of this crate). It includes rayon
//! spin-wait on the multi-thread pool, which is real CPU the tree also pays.
//!
//! NOT A LEVER. Read-only timing; nothing here is reachable from a verdict
//! path.

use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::Array2;
use ny_propagate::margin_row::bounds::{
    compose_viay, head_gates, margin_seed, row_dots, MarginBatch, YBox,
};
use ny_propagate::margin_row::engine::{
    domain_gates, BackwardEngine, Collect, DomainGates, LaneDir, PassOut, Seed,
};
use ny_propagate::margin_row::net::{conv_apply_backward, TwinOp};
use ny_propagate::margin_row::ErrLane;
use ny_propagate::margin_row::{RootGates, RoundMode, TwinNet};

const TAG: &str = "[colscale2]";

/// Parsed probe arguments.
struct Config {
    onnx: PathBuf,
    vnnlib: PathBuf,
    threads: Vec<usize>,
    cols: Vec<usize>,
    reps: usize,
    seed: u64,
    depth: usize,
    ks: Vec<usize>,
    nfks: Vec<usize>,
    arms: Vec<String>,
}

fn parse_list(key: &str, value: &str) -> anyhow::Result<Vec<usize>> {
    let out: Vec<usize> = value
        .split('+')
        .map(|v| {
            v.trim()
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("col-scaling-v2: bad {key} entry {v:?}: {e}"))
        })
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(
        !out.is_empty() && out.iter().all(|&v| v > 0),
        "col-scaling-v2: {key} must be a non-empty '+' list of positive integers"
    );
    Ok(out)
}

impl Config {
    fn parse(args: &str) -> anyhow::Result<Self> {
        let mut onnx = None;
        let mut vnnlib = None;
        let mut cfg = Self {
            onnx: PathBuf::new(),
            vnnlib: PathBuf::new(),
            threads: vec![1, 20],
            cols: vec![1, 2, 3, 4, 8, 16, 64, 192],
            reps: 5,
            seed: 1,
            depth: 20,
            ks: vec![4, 8, 20],
            nfks: vec![1, 3],
            arms: vec!["pass".into(), "absw".into(), "stack".into()],
        };
        for part in args.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (key, value) = part.split_once('=').ok_or_else(|| {
                anyhow::anyhow!("col-scaling-v2: argument {part:?} is not key=value")
            })?;
            match key {
                "onnx" => onnx = Some(PathBuf::from(value)),
                "vnnlib" => vnnlib = Some(PathBuf::from(value)),
                "threads" => cfg.threads = parse_list(key, value)?,
                "cols" => cfg.cols = parse_list(key, value)?,
                "reps" => cfg.reps = parse_list(key, value)?[0],
                "seed" => cfg.seed = parse_list(key, value)?[0] as u64,
                "depth" => cfg.depth = parse_list(key, value)?[0],
                "k" => cfg.ks = parse_list(key, value)?,
                "nfk" => cfg.nfks = parse_list(key, value)?,
                "arms" => {
                    cfg.arms = value.split('+').map(|a| a.trim().to_string()).collect();
                    for arm in &cfg.arms {
                        anyhow::ensure!(
                            matches!(arm.as_str(), "pass" | "absw" | "stack" | "eel"),
                            "col-scaling-v2: unknown arm {arm:?} (pass, absw, stack, eel)"
                        );
                    }
                }
                _ => anyhow::bail!(
                    "col-scaling-v2: unknown key {key:?} (onnx, vnnlib, threads, cols, reps, \
                     seed, depth, k, nfk, arms)"
                ),
            }
        }
        cfg.onnx =
            onnx.ok_or_else(|| anyhow::anyhow!("col-scaling-v2: onnx=<abs path> required"))?;
        cfg.vnnlib =
            vnnlib.ok_or_else(|| anyhow::anyhow!("col-scaling-v2: vnnlib=<abs path> required"))?;
        for path in [&cfg.onnx, &cfg.vnnlib] {
            anyhow::ensure!(
                path.is_absolute() && path.is_file(),
                "col-scaling-v2: {} is not an existing absolute file path",
                path.display()
            );
        }
        Ok(cfg)
    }

    fn arm(&self, name: &str) -> bool {
        self.arms.iter().any(|a| a == name)
    }
}

/// Process CPU nanoseconds (`CLOCK_PROCESS_CPUTIME_ID`).
#[cfg(unix)]
fn process_cpu_ns() -> Option<u64> {
    let ts = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
    u64::try_from(ts.tv_sec)
        .ok()?
        .checked_mul(1_000_000_000)?
        .checked_add(u64::try_from(ts.tv_nsec).ok()?)
}

#[cfg(not(unix))]
fn process_cpu_ns() -> Option<u64> {
    None
}

/// Deterministic xorshift for the shuffled R order (printed, so reproducible).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self((seed ^ 0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

/// Median wall ms and median process-CPU ms over `reps` timed calls (one
/// untimed warm call first). CPU is NaN when the clock is unavailable.
fn measure(reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    f();
    let mut walls = Vec::with_capacity(reps);
    let mut cpus = Vec::with_capacity(reps);
    for _ in 0..reps.max(1) {
        let c0 = process_cpu_ns();
        let t0 = Instant::now();
        f();
        walls.push(t0.elapsed().as_secs_f64() * 1e3);
        cpus.push(match (c0, process_cpu_ns()) {
            (Some(a), Some(b)) => b.saturating_sub(a) as f64 / 1e6,
            _ => f64::NAN,
        });
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    (median(walls), median(cpus))
}

/// `depth` trunk splits spread round-robin over the layers; `k` offsets the
/// positions (and signs) so each probe domain has DISTINCT gates. A split is
/// `(layer, position-in-unst, sign)`, the convention `domain_gates` expects.
fn deep_splits(root: &RootGates, depth: usize, k: usize) -> Vec<(usize, usize, i8)> {
    let mut splits = Vec::new();
    let mut taken = vec![0usize; root.layers.len()];
    'outer: loop {
        let before = splits.len();
        for (li, lg) in root.layers.iter().enumerate() {
            let n = lg.unst.len();
            if taken[li] < n {
                let pos = (taken[li] + k * 7) % n;
                taken[li] += 1;
                let sign = if (splits.len() + k).is_multiple_of(2) {
                    1i8
                } else {
                    -1i8
                };
                splits.push((li, pos, sign));
                if splits.len() >= depth {
                    break 'outer;
                }
            }
        }
        if splits.len() == before {
            break;
        }
    }
    splits
}

/// Tile the real margin seed's columns out to width `r` (column values cannot
/// change the cost — every skip tests a WEIGHT — but real magnitudes keep the
/// certified error lane representative). Identity-like when nothing failed.
fn tile_seed(base: &Seed, r: usize) -> Seed {
    let n_y = base.s.nrows();
    let nf = base.s.ncols();
    let mut s = Array2::<f64>::zeros((n_y, r));
    let mut e = base.e.as_ref().map(|_| Array2::<f64>::zeros((n_y, r)));
    for c in 0..r {
        for j in 0..n_y {
            if nf == 0 {
                s[[j, c]] = f64::from(u8::from(j == c % n_y.max(1)));
            } else {
                s[[j, c]] = base.s[[j, c % nf]];
                if let (Some(dst), Some(src)) = (e.as_mut(), base.e.as_ref()) {
                    dst[[j, c]] = src[[j, c % nf]];
                }
            }
        }
    }
    Seed { s, e }
}

/// Horizontal concatenation of `k` copies of `one` (the stacked seed).
fn repeat_seed(one: &Seed, k: usize) -> Seed {
    let n_y = one.s.nrows();
    let w = one.s.ncols();
    let mut s = Array2::<f64>::zeros((n_y, w * k));
    let mut e = one.e.as_ref().map(|_| Array2::<f64>::zeros((n_y, w * k)));
    for b in 0..k {
        for c in 0..w {
            for j in 0..n_y {
                s[[j, b * w + c]] = one.s[[j, c]];
                if let (Some(dst), Some(src)) = (e.as_mut(), one.e.as_ref()) {
                    dst[[j, b * w + c]] = src[[j, c]];
                }
            }
        }
    }
    Seed { s, e }
}

/// One rounding mode's evaluation frame: its own root, the real root-fail
/// margin seed, and the deep probe domains.
struct Frame {
    mode: RoundMode,
    root: RootGates,
    seed: Seed,
    nf: usize,
}

fn frame(
    net: &TwinNet,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    mode: RoundMode,
) -> anyhow::Result<Frame> {
    let root = RootGates::build(net, lo, hi, mode, None)
        .map_err(|e| anyhow::anyhow!("col-scaling-v2: root build ({mode:?}): {e}"))?;
    let eng = BackwardEngine::new(net, &root);
    let (al, au) = eng
        .y_rows(None)
        .map_err(|e| anyhow::anyhow!("col-scaling-v2: y_rows: {e}"))?;
    let ybox = YBox::from_rows(&eng, &al, &au);
    let mb_all = MarginBatch::new(net, t, adv)
        .map_err(|e| anyhow::anyhow!("col-scaling-v2: margin batch: {e}"))?;
    let gates = head_gates(&ybox, root.mode);
    let ms_all = margin_seed(&mb_all, &gates, &ybox, root.mode);
    let ald = row_dots(&root, &al);
    let aud = row_dots(&root, &au);
    let m2v = compose_viay(&eng, &mb_all, &gates, &al, &au, &ald, &aud, root.mode);
    let fail: Vec<usize> = (0..adv.len())
        .filter(|&k| ms_all.m1[k].max(m2v[k]) < 0.0)
        .map(|k| adv[k])
        .collect();
    // An all-closed root still yields a probe seed (the full adv set), so the
    // cost curve is measurable on any row.
    let classes: &[usize] = if fail.is_empty() { adv } else { &fail };
    let mb = MarginBatch::new(net, t, classes)
        .map_err(|e| anyhow::anyhow!("col-scaling-v2: fail-set margin batch: {e}"))?;
    let seed = margin_seed(&mb, &gates, &ybox, root.mode).seed;
    let nf = fail.len();
    Ok(Frame {
        mode,
        root,
        seed,
        nf,
    })
}

/// Entry point (selector `col-scaling-v2[,key=value...]`).
pub(crate) fn run(args: &str) -> anyhow::Result<()> {
    let cfg = Config::parse(args)?;
    let spec = super::extract_twin_spec(&cfg.onnx).ok_or_else(|| {
        anyhow::anyhow!(
            "col-scaling-v2: not a twin-wall net: {}",
            cfg.onnx.display()
        )
    })?;
    let (lo, hi, t, adv) = super::parse_vnnlib_robustness(&cfg.vnnlib).ok_or_else(|| {
        anyhow::anyhow!(
            "col-scaling-v2: not a robustness property: {}",
            cfg.vnnlib.display()
        )
    })?;
    let net =
        TwinNet::compile(&spec).map_err(|e| anyhow::anyhow!("col-scaling-v2: compile: {e}"))?;
    let mut rng = Rng::new(cfg.seed);
    let mut order = cfg.cols.clone();
    rng.shuffle(&mut order);
    eprintln!(
        "{TAG} config onnx={} vnnlib={} threads={:?} reps={} seed={} depth={} k={:?} nfk={:?} \
         arms={:?} cols={:?} shuffled_order={order:?}",
        cfg.onnx.display(),
        cfg.vnnlib.display(),
        cfg.threads,
        cfg.reps,
        cfg.seed,
        cfg.depth,
        cfg.ks,
        cfg.nfks,
        cfg.arms,
        cfg.cols,
    );
    let convs: Vec<&ny_propagate::margin_row::net::ConvOp> = net
        .ops
        .iter()
        .filter_map(|op| match op {
            TwinOp::Conv(c) => Some(&**c),
            _ => None,
        })
        .collect();
    let widest = net.tsize.iter().copied().max().unwrap_or(0);
    let outward = frame(&net, &lo, &hi, t, &adv, RoundMode::Outward)?;
    let parity = frame(&net, &lo, &hi, t, &adv, RoundMode::Parity)?;
    let n_unst: usize = outward.root.layers.iter().map(|l| l.unst.len()).sum();
    eprintln!(
        "{TAG} net n_in={} n_y={} trunk_relus={} unstable={n_unst} convs={} widest_tensor={widest} \
         root_fail_nf={} global_threads={}",
        net.n_in,
        net.n_y,
        outward.root.layers.len(),
        convs.len(),
        outward.nf,
        rayon::current_num_threads()
    );
    let scored_collect = Collect {
        unst_abs: true,
        rows: None,
        unst_rows: false,
    };
    for &threads in &cfg.threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| anyhow::anyhow!("col-scaling-v2: {threads}-thread pool: {e}"))?;
        // ---------------- arm: pass (Outward vs Parity) ----------------
        if cfg.arm("pass") {
            let mut cpu_by: Vec<(usize, f64, f64)> = Vec::new(); // (R, outward, parity)
            for &r in &order {
                let mut row = (r, f64::NAN, f64::NAN);
                for fr in [&outward, &parity] {
                    let eng = BackwardEngine::new(&net, &fr.root);
                    let dom = domain_gates(&fr.root, &deep_splits(&fr.root, cfg.depth, 0));
                    let seed = tile_seed(&fr.seed, r);
                    let mut failed = None;
                    let (wall, cpu) = pool.install(|| {
                        measure(cfg.reps, || {
                            if let Err(e) = eng.run_collect(
                                &seed,
                                Some(&dom),
                                LaneDir::Lower,
                                None,
                                scored_collect,
                            ) {
                                failed = Some(e.to_string());
                            }
                        })
                    });
                    if let Some(e) = failed {
                        eprintln!(
                            "{TAG} pass threads={threads} mode={:?} R={r} ERR {e}",
                            fr.mode
                        );
                        continue;
                    }
                    eprintln!(
                        "{TAG} pass threads={threads} mode={:?} R={r} wall_ms={wall:.3} \
                         cpu_ms={cpu:.3} cpu_per_col_ms={:.4} cpu_over_wall={:.2}",
                        fr.mode,
                        cpu / r as f64,
                        cpu / wall
                    );
                    if matches!(fr.mode, RoundMode::Outward) {
                        row.1 = cpu;
                    } else {
                        row.2 = cpu;
                    }
                }
                cpu_by.push(row);
            }
            cpu_by.sort_by_key(|row| row.0);
            for (r, out_cpu, par_cpu) in cpu_by {
                eprintln!(
                    "{TAG} errlane threads={threads} R={r} outward_cpu_ms={out_cpu:.3} \
                     parity_cpu_ms={par_cpu:.3} outward_over_parity={:.3} errlane_share={:.3}",
                    out_cpu / par_cpu,
                    1.0 - par_cpu / out_cpu
                );
            }
        }
        // ---------------- arm: eel (#mr-errlane, plan B3) ----------------
        if cfg.arm("eel") {
            let dom = domain_gates(&outward.root, &deep_splits(&outward.root, cfg.depth, 0));
            let lanes = [ErrLane::PerCol, ErrLane::Envelope, ErrLane::PerChannel];
            let mut rows: Vec<(usize, String)> = Vec::new();
            for &r in &order {
                let seed = tile_seed(&outward.seed, r);
                let mut cpu = [f64::NAN; 3];
                let mut outs: [Option<PassOut>; 3] = [None, None, None];
                for (i, lane) in lanes.into_iter().enumerate() {
                    let eng = BackwardEngine::new(&net, &outward.root).with_err_lane(lane);
                    let mut failed = None;
                    let mut last = None;
                    let (wall, c) = pool.install(|| {
                        measure(cfg.reps, || {
                            match eng.run_collect(
                                &seed,
                                Some(&dom),
                                LaneDir::Lower,
                                None,
                                scored_collect,
                            ) {
                                Ok(out) => last = Some(out),
                                Err(e) => failed = Some(e.to_string()),
                            }
                        })
                    });
                    if let Some(e) = failed {
                        eprintln!(
                            "{TAG} eel threads={threads} lane={} R={r} ERR {e}",
                            lane.token()
                        );
                        continue;
                    }
                    eprintln!(
                        "{TAG} eel threads={threads} lane={} R={r} wall_ms={wall:.3} \
                         cpu_ms={c:.3} cpu_per_col_ms={:.4}",
                        lane.token(),
                        c / r as f64
                    );
                    cpu[i] = c;
                    outs[i] = last;
                }
                let eng = BackwardEngine::new(&net, &outward.root);
                let Some(col) = outs[0].as_ref() else {
                    continue;
                };
                let bc = eng.concretize_lower(col);
                let loosen = |env: Option<&PassOut>| -> String {
                    let Some(env) = env else {
                        return "na:na:na:na".into();
                    };
                    let be = eng.concretize_lower(env);
                    let (mut max_l, mut max_rel, mut min_pos, mut lost) =
                        (0.0f64, 0.0f64, f64::INFINITY, 0usize);
                    for (c, e) in bc.iter().zip(&be) {
                        let l = (c - e).max(0.0);
                        max_l = max_l.max(l);
                        if *c > 0.0 {
                            min_pos = min_pos.min(*c);
                            max_rel = max_rel.max(l / c);
                            lost += usize::from(*e <= 0.0);
                        }
                    }
                    format!("{max_l:.3e}:{max_rel:.3e}:{min_pos:.6e}:{lost}")
                };
                rows.push((
                    r,
                    format!(
                        "{TAG} eel-summary threads={threads} R={r} env_engaged={} \
                     off_cpu_ms={:.3} env_cpu_ms={:.3} pc_cpu_ms={:.3} env_over_off={:.3} \
                     pc_over_off={:.3} env_loosen={} pc_loosen={} \
                     (loosen = max_loosen:max_rel:min_pos:pos_lost)",
                        u8::from(r >= ny_propagate::margin_row::errlane::ENV_MIN_COLS),
                        cpu[0],
                        cpu[1],
                        cpu[2],
                        cpu[1] / cpu[0],
                        cpu[2] / cpu[0],
                        loosen(outs[1].as_ref()),
                        loosen(outs[2].as_ref()),
                    ),
                ));
            }
            rows.sort_by_key(|row| row.0);
            for (_, row) in rows {
                eprintln!("{row}");
            }
        }
        // ---------------- arm: absw (conv backward alone at r = 1) ----------------
        if cfg.arm("absw") && !convs.is_empty() {
            let srcs: Vec<Array2<f64>> = convs
                .iter()
                .map(|c| {
                    let rows = c.oshape.0 * c.oshape.1 * c.oshape.2;
                    Array2::from_shape_fn((rows, 1), |(i, _)| ((i % 13) as f64 - 6.0) / 8.0)
                })
                .collect();
            for abs_w in [false, true] {
                let mut dsts: Vec<Array2<f64>> = convs
                    .iter()
                    .map(|c| Array2::<f64>::zeros((c.ishape.0 * c.ishape.1 * c.ishape.2, 1)))
                    .collect();
                let (wall, cpu) = pool.install(|| {
                    measure(cfg.reps, || {
                        for ((c, src), dst) in convs.iter().zip(&srcs).zip(dsts.iter_mut()) {
                            conv_apply_backward(c, src, dst, abs_w);
                        }
                    })
                });
                eprintln!(
                    "{TAG} absw threads={threads} r=1 weights={} convs={} wall_ms={wall:.3} \
                     cpu_ms={cpu:.3} (every conv's backward kernel once, summed)",
                    if abs_w { "abs" } else { "signed" },
                    convs.len()
                );
            }
        }
        // ---------------- arm: stack (K concurrent vs one stacked) ----------------
        if cfg.arm("stack") {
            let eng = BackwardEngine::new(&net, &outward.root);
            for &k in &cfg.ks {
                let doms: Vec<DomainGates> = (0..k)
                    .map(|i| domain_gates(&outward.root, &deep_splits(&outward.root, cfg.depth, i)))
                    .collect();
                for &nf in &cfg.nfks {
                    let one = tile_seed(&outward.seed, nf);
                    let stacked_seed = repeat_seed(&one, k);
                    let blocks: Vec<(Range<usize>, &DomainGates)> = doms
                        .iter()
                        .enumerate()
                        .map(|(i, d)| (i * nf..(i + 1) * nf, d))
                        .collect();
                    // Default collection in BOTH arms (the stacked pass's
                    // per-block `unst_abs` split is B5-lite's job), so the
                    // comparison is like-for-like.
                    let plain = Collect::default();
                    let mut conc_low: Vec<Vec<f64>> = Vec::new();
                    let mut stk_low: Vec<f64> = Vec::new();
                    let mut failed = None;
                    let (cw, cc) = pool.install(|| {
                        measure(cfg.reps, || {
                            use rayon::prelude::*;
                            let outs: Vec<_> = doms
                                .par_iter()
                                .map(|d| {
                                    eng.run_collect(&one, Some(d), LaneDir::Lower, None, plain)
                                })
                                .collect();
                            conc_low = outs
                                .iter()
                                .filter_map(|o| o.as_ref().ok().map(|p| eng.concretize_lower(p)))
                                .collect();
                            if let Some(Err(e)) = outs.into_iter().find(Result::is_err) {
                                failed = Some(e.to_string());
                            }
                        })
                    });
                    let (sw, sc) = pool.install(|| {
                        measure(cfg.reps, || {
                            match eng.probe_domain_stacked_collect(
                                &stacked_seed,
                                &blocks,
                                LaneDir::Lower,
                                plain,
                            ) {
                                Ok(p) => stk_low = eng.concretize_lower(&p),
                                Err(e) => failed = Some(e.to_string()),
                            }
                        })
                    });
                    if let Some(e) = failed {
                        eprintln!("{TAG} stack threads={threads} K={k} nf={nf} ERR {e}");
                        continue;
                    }
                    let flat: Vec<f64> = conc_low.concat();
                    let bitident = flat.len() == stk_low.len()
                        && flat
                            .iter()
                            .zip(&stk_low)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    eprintln!(
                        "{TAG} stack threads={threads} K={k} nf={nf} concurrent_wall_ms={cw:.3} \
                         concurrent_cpu_ms={cc:.3} stacked_wall_ms={sw:.3} stacked_cpu_ms={sc:.3} \
                         stacked_over_concurrent_cpu={:.3} stacked_over_concurrent_wall={:.3} \
                         bitident={}",
                        sc / cc,
                        sw / cw,
                        u8::from(bitident)
                    );
                }
            }
        }
    }
    eprintln!(
        "{TAG} gates (plan §3): B3/EEL reads errlane outward_over_parity at R>=64 threads=1 \
         (>=1.5 build, <1.4 drop); B5 reads stack stacked_over_concurrent_cpu at K>=8 nf<=3 \
         (<=0.35 build); the absw r=1 lines are the narrow-width floor's direct target"
    );
    // #narrow-r (B1(d)): the floor's own receipt for this process — absent on
    // a binary built before it, `engaged=0` when it never had to split a job.
    eprintln!(
        "{TAG} {}",
        ny_propagate::margin_row::net::narrow_r_stop_line()
    );
    eprintln!("{TAG} DONE");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_list, repeat_seed, tile_seed, Config, Rng};
    use ndarray::Array2;
    use ny_propagate::margin_row::engine::Seed;

    /// `col_scaling_v2_args_parse_and_refuse`: the selector grammar, including
    /// the refusals that keep a typo from silently measuring the wrong thing.
    #[test]
    fn col_scaling_v2_args_parse_and_refuse() {
        assert_eq!(parse_list("cols", "1+2+64").expect("list"), vec![1, 2, 64]);
        assert!(parse_list("cols", "1+0").is_err(), "zero width refused");
        assert!(parse_list("cols", "").is_err(), "empty list refused");
        assert!(Config::parse("").is_err(), "onnx/vnnlib are required");
        assert!(
            Config::parse(",onnx=relative.onnx,vnnlib=relative.vnnlib").is_err(),
            "relative / missing paths refused"
        );
        assert!(Config::parse(",bogus=1").is_err(), "unknown key refused");
    }

    /// `col_scaling_v2_shuffle_is_a_permutation`: the shuffled order is a
    /// permutation of the requested widths and is reproducible from the seed.
    #[test]
    fn col_scaling_v2_shuffle_is_a_permutation() {
        let base = vec![1usize, 2, 3, 4, 8, 16, 64, 192];
        let mut a = base.clone();
        Rng::new(7).shuffle(&mut a);
        let mut b = base.clone();
        Rng::new(7).shuffle(&mut b);
        assert_eq!(a, b, "same seed, same order");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, base, "a permutation, nothing lost or duplicated");
    }

    /// `col_scaling_v2_seed_tiling_and_stacking`: stacked block `b` is an exact
    /// copy of the per-domain seed, so the stacked arm prices the same columns.
    #[test]
    fn col_scaling_v2_seed_tiling_and_stacking() {
        let base = Seed {
            s: Array2::from_shape_fn((3, 2), |(j, c)| (j * 10 + c) as f64),
            e: Some(Array2::from_shape_fn((3, 2), |(j, c)| {
                (j + c) as f64 * 1e-9
            })),
        };
        let one = tile_seed(&base, 3);
        assert_eq!(one.s[[2, 2]].to_bits(), base.s[[2, 0]].to_bits());
        let stk = repeat_seed(&one, 4);
        assert_eq!(stk.s.ncols(), 12);
        for b in 0..4 {
            for c in 0..3 {
                for j in 0..3 {
                    assert_eq!(stk.s[[j, b * 3 + c]].to_bits(), one.s[[j, c]].to_bits());
                    let (se, oe) = (stk.e.as_ref().expect("e"), one.e.as_ref().expect("e"));
                    assert_eq!(se[[j, b * 3 + c]].to_bits(), oe[[j, c]].to_bits());
                }
            }
        }
    }
}
