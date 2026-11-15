// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#wide-fd-search` (ENGINE_PROGRAM M4), DEVICE side: the KNOWN-ANSWER tests
//! of ny-cuda's level-search kernels against the CPU f32 search of ny-bbe,
//! which is their REFERENCE. ny-cli is the only crate that sees both backends
//! (ny-cuda may not depend on ny-bbe: `crate_dag_ratchet`).
//!
//! The contract is BIT IDENTITY of every output the tier reads — each node's
//! value and `S` at the four points (NaN = not scored), its clamped and far
//! multipliers and its children's warm start — plus the untrusted counters
//! (nodes, evaluations, refusals). Two canonicalisations only: a NaN is a
//! NaN, and the sign of a zero is not an output (the CPU adds `0 * a` to a
//! lane whose multiplier is 0 where a kernel lane skips the row).
//!
//! * `cuda_fd_level_matches_cpu_f32_bit_for_bit_on_synthetic_levels` — random
//!   and dyadic problems; widths that are and are not multiples of the column
//!   tile; depths past the kernels' 64-row staging; node counts that are not
//!   multiples of a CTA; sub-depth chunks; dirty (NaN / inf / negative)
//!   multipliers; overflowing (refused) nodes; margins that make every node
//!   clearly positive (far-first skips) or none;
//! * `cuda_fd_level_matches_cpu_f32_on_tiny_and_cifar_shaped_levels` — the
//!   generator counts and depths of the scored tiny / cifar states;
//! * `cuda_fd_level_is_launch_and_host_thread_independent` — the device
//!   answer does not move with the CPU backend's thread count (it is the same
//!   answer) nor with the lane's chunk cut;
//! * `cuda_fd_kat_comparator_catches_a_one_ulp_mutation` — the MUTATION PIN of
//!   the comparator itself (device-free: it runs on every host).
//!
//! GATING. On a host that advertises a CUDA device every device test must
//! pass; on any other host each prints `SKIP=VOID` and PROVES NOTHING (it is
//! not a pass). The lane-level tests (the P1 trees, the sat class, the registered
//! factory) are in `frozen_dual_bab_tests`.

use ny_core::search::{
    DeviceLease, FdSearchProblem, LevelChunk, LevelOut, LevelStats, SearchBackend, FD_CLAMPED,
    FD_FAR, FD_FLOOR, FD_POINTS, FD_WARM,
};

const BIG: DeviceLease = DeviceLease { bytes: 1 << 34 };

/// A device the tests must use, or `None` after printing `SKIP=VOID`.
pub(super) fn search_device(test: &str) -> Option<ny_cuda::search::CudaSearchBackend> {
    let library = ny_cuda::CudaGemmEngine::runtime_driver_library_present();
    let devices = if library {
        ny_cuda::CudaGemmEngine::runtime_device_count()
    } else {
        None
    };
    if !(library && devices.is_some_and(|c| c > 0)) {
        eprintln!(
            "[cuda-fd-kat] {test}: SKIP=VOID (libcuda={library}, devices={devices:?}) — no CUDA \
             device on this host; this run proved NOTHING about the device search"
        );
        return None;
    }
    let backend = ny_cuda::search::CudaSearchBackend::new();
    // The device is advertised: a refusal to open it is a failure, not a skip.
    let probe = FdSearchProblem {
        n: 1,
        depth: 1,
        a: vec![0.5],
        enc: vec![0],
        rho: vec![[1.0, -1.0]],
        c0s: vec![[0.0, 0.0]],
        b0: vec![[0.0, 0.0]],
        b1: vec![[0.0, 0.0]],
        row_abs: vec![0.5],
        d: vec![0.25],
        c0: 1.0,
        abs_d: 0.25,
    };
    if let Err(why) = backend.upload_fd(&probe, &BIG) {
        panic!(
            "{test}: this host advertises a CUDA device ({devices:?}) but the device search \
             refused it: {why} (unavailable: {:?})",
            backend.unavailable()
        );
    }
    Some(backend)
}

/// splitmix64.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, k: u64) -> u64 {
        self.next() % k.max(1)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }

    fn dyadic(&mut self, lo: i64, hi: i64, den: f64) -> f32 {
        let k = lo + self.below((hi - lo + 1) as u64) as i64;
        (k as f64 / den) as f32
    }
}

/// A random search copy (`exact`: dyadic entries), as ny-bbe's fd tests make.
pub(super) fn problem(n: usize, depth: usize, seed: u64, exact: bool) -> FdSearchProblem {
    let mut rng = Mix(seed ^ 0x5EED_F00D);
    let depth = depth.min(n);
    let mut cols: Vec<u32> = (0..n as u32).collect();
    for i in 0..depth {
        let j = i + rng.below((n - i) as u64) as usize;
        cols.swap(i, j);
    }
    let a: Vec<f32> = (0..depth * n)
        .map(|_| {
            if rng.below(5) == 0 {
                0.0
            } else if exact {
                rng.dyadic(-16, 16, 16.0)
            } else {
                rng.range(-1.0, 1.0) as f32
            }
        })
        .collect();
    let row_abs: Vec<f32> = (0..depth)
        .map(|i| a[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum::<f32>())
        .collect();
    let mut pair = |lo: f64, hi: f64, den: f64| -> [f32; 2] {
        if exact {
            [
                rng.dyadic((lo * den) as i64, (hi * den) as i64, den),
                rng.dyadic((lo * den) as i64, (hi * den) as i64, den),
            ]
        } else {
            [rng.range(lo, hi) as f32, rng.range(lo, hi) as f32]
        }
    };
    let rho: Vec<[f32; 2]> = (0..depth).map(|_| pair(-2.0, 2.0, 8.0)).collect();
    let c0s: Vec<[f32; 2]> = (0..depth).map(|_| pair(-0.5, 0.0, 32.0)).collect();
    let b0: Vec<[f32; 2]> = (0..depth).map(|_| pair(0.0, 3.0, 16.0)).collect();
    let b1: Vec<[f32; 2]> = (0..depth).map(|_| pair(0.0, 3.0, 16.0)).collect();
    let d: Vec<f32> = (0..n)
        .map(|_| {
            if exact {
                rng.dyadic(-16, 16, 16.0)
            } else {
                rng.range(-1.0, 1.0) as f32
            }
        })
        .collect();
    let abs_d: f32 = d.iter().map(|v| v.abs()).sum();
    let c0 = if exact {
        (0.5 * abs_d).floor()
    } else {
        (0.5 * f64::from(abs_d)) as f32
    };
    FdSearchProblem {
        n,
        depth,
        a,
        enc: cols[..depth].to_vec(),
        rho,
        c0s,
        b0,
        b1,
        row_abs,
        d,
        c0,
        abs_d,
    }
}

/// A random level of `count` nodes of `depth`.
pub(super) fn level(
    count: usize,
    depth: usize,
    seed: u64,
    exact: bool,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut rng = Mix(seed ^ 0x1E7E_1000);
    let mut sides = Vec::with_capacity(count * depth);
    let mut l0 = Vec::with_capacity(count * depth);
    let mut l1 = Vec::with_capacity(count * depth);
    for _ in 0..count * depth {
        sides.push(u8::from(rng.below(2) == 1));
        let mut m = || {
            if rng.below(4) == 0 {
                0.0
            } else if exact {
                rng.dyadic(0, 16, 8.0)
            } else {
                rng.range(0.0, 2.0) as f32
            }
        };
        l0.push(m());
        l1.push(m());
    }
    (sides, l0, l1)
}

fn chunk<'a>(depth: usize, sides: &'a [u8], l0: &'a [f32], l1: &'a [f32]) -> LevelChunk<'a> {
    LevelChunk {
        depth,
        count: sides.len() / depth,
        sides,
        l0,
        l1,
        deadline: None,
    }
}

/// One level on `backend`: its outputs and untrusted counters.
fn run(
    backend: &dyn SearchBackend,
    p: &FdSearchProblem,
    c: &LevelChunk<'_>,
) -> (LevelOut, LevelStats) {
    let h = backend
        .upload_fd(p, &BIG)
        .unwrap_or_else(|why| panic!("{}: upload refused: {why}", backend.name()));
    let mut out = LevelOut::default();
    let stats = backend
        .fd_level(&h, c, &mut out)
        .unwrap_or_else(|why| panic!("{}: fd_level refused: {why}", backend.name()))
        .into_hint()
        .into_ingest();
    (out, stats)
}

/// A NaN is a NaN; a zero is a zero (its sign is not an output).
fn canon(x: f32) -> u32 {
    if x.is_nan() {
        0x7FC0_0000
    } else if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}

/// Coverage of the behaviours a KAT must have exercised (else it is VOID).
#[derive(Debug, Default)]
struct Coverage {
    nodes: usize,
    refused: usize,
    far_moved: usize,
    skipped: usize,
    scored_c: usize,
    stepped: usize,
}

/// Compare the two outputs field by field; panic naming the first mismatch.
fn assert_same(tag: &str, cpu: &LevelOut, dev: &LevelOut, l0: &[f32], cov: &mut Coverage) {
    let (c, g) = (cpu.hint().into_ingest(), dev.hint().into_ingest());
    assert!(
        c.well_formed() && g.well_formed(),
        "{tag}: malformed output"
    );
    assert_eq!(
        (c.count(), c.depth()),
        (g.count(), g.depth()),
        "{tag}: shape"
    );
    let d = c.depth();
    for k in 0..c.count() {
        for pt in 0..FD_POINTS {
            for (what, x, y) in [
                ("value", c.value(k, pt), g.value(k, pt)),
                ("S", c.scale(k, pt), g.scale(k, pt)),
            ] {
                assert_eq!(
                    canon(x),
                    canon(y),
                    "{tag}: node {k} point {pt} {what}: cpu_f32 {x:e} ({:#010x}) vs cuda {y:e} \
                     ({:#010x})",
                    x.to_bits(),
                    y.to_bits()
                );
            }
        }
        for (what, (x0, x1), (y0, y1)) in [
            ("clamped", c.clamped(k), g.clamped(k)),
            ("far", c.far(k), g.far(k)),
            ("next", c.next(k), g.next(k)),
        ] {
            for i in 0..d {
                for (side, x, y) in [("l0", x0[i], y0[i]), ("l1", x1[i], y1[i])] {
                    assert_eq!(
                        canon(x),
                        canon(y),
                        "{tag}: node {k} {what}.{side}[{i}]: cpu_f32 {x:e} vs cuda {y:e}"
                    );
                }
            }
        }
        cov.nodes += 1;
        let v = |pt| c.value(k, pt);
        if v(FD_WARM).is_nan() {
            cov.refused += 1;
            continue;
        }
        cov.far_moved += usize::from(v(FD_FAR).is_finite() && v(FD_FAR) != v(FD_WARM));
        cov.skipped += usize::from(v(FD_FLOOR).is_nan() || v(FD_CLAMPED).is_nan());
        cov.scored_c += usize::from(v(FD_FLOOR).is_finite() && v(FD_CLAMPED).is_finite());
        cov.stepped += usize::from(c.next(k).0 != &l0[k * d..(k + 1) * d]);
    }
}

/// Run one chunk on both backends and compare everything.
fn kat(
    tag: &str,
    dev: &ny_cuda::search::CudaSearchBackend,
    p: &FdSearchProblem,
    depth: usize,
    lv: &(Vec<u8>, Vec<f32>, Vec<f32>),
    cov: &mut Coverage,
) {
    let cpu = ny_bbe::CpuF32Backend::with_threads(3);
    let c = chunk(depth, &lv.0, &lv.1, &lv.2);
    let (co, cs) = run(&cpu, p, &c);
    let (go, gs) = run(dev, p, &c);
    assert_same(tag, &co, &go, &lv.1, cov);
    assert_eq!(
        (cs.nodes, cs.evals, cs.refused),
        (gs.nodes, gs.evals, gs.refused),
        "{tag}: the untrusted counters differ"
    );
}

#[test]
fn cuda_fd_level_matches_cpu_f32_bit_for_bit_on_synthetic_levels() {
    let test = "cuda_fd_level_matches_cpu_f32_bit_for_bit_on_synthetic_levels";
    let Some(dev) = search_device(test) else {
        return;
    };
    let mut cov = Coverage::default();
    // (n, depth, count, seed): widths below / at / above one tile, depths past
    // the 64-row staging, counts that are not multiples of a CTA (8).
    let shapes = [
        (1usize, 1usize, 3usize, 1u64),
        (37, 5, 19, 2),
        (255, 3, 8, 3),
        (256, 7, 16, 4),
        (257, 12, 33, 5),
        (700, 30, 41, 6),
        (2049, 28, 26, 7),
        (600, 70, 17, 8),
        (300, 130, 9, 9),
    ];
    for &(n, depth, count, seed) in &shapes {
        for exact in [false, true] {
            let p = problem(n, depth, seed, exact);
            let depth = p.depth;
            for d in [1, depth / 2 + 1, depth] {
                let lv = level(count, d, seed ^ (d as u64) << 8, exact);
                kat(
                    &format!("n={n} depth={depth} d={d} exact={exact}"),
                    &dev,
                    &p,
                    d,
                    &lv,
                    &mut cov,
                );
            }
        }
    }
    // Dirty multipliers (sanitised to 0 by both), and overflowing nodes
    // (refused by both: no value, the parent's point proposed).
    let p = problem(333, 9, 21, false);
    let (sides, mut l0, mut l1) = level(29, 9, 21, false);
    for (k, bad) in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -3.0,
        -0.0,
        1e-45,
    ]
    .into_iter()
    .enumerate()
    {
        l0[k * 9 + k % 9] = bad;
        l1[(k + 6) * 9 + 1] = bad;
    }
    for m in &mut l0[20 * 9..21 * 9] {
        *m = 3.0e38;
    }
    for m in &mut l1[22 * 9..23 * 9] {
        *m = f32::MAX;
    }
    kat("dirty+overflow", &dev, &p, 9, &(sides, l0, l1), &mut cov);
    // Every node clearly positive (the clamped / floor points skipped), then
    // none.
    for (c0, tag) in [(1e5f32, "all-positive"), (-1e5, "all-negative")] {
        let mut p = problem(150, 6, 71, false);
        p.c0 = c0;
        let lv = level(2 * 8 + 3, 6, 71, false);
        kat(tag, &dev, &p, 6, &lv, &mut cov);
    }
    eprintln!("[cuda-fd-kat] {test}: {cov:?}");
    assert!(
        cov.nodes > 1_000
            && cov.refused > 0
            && cov.far_moved > 100
            && cov.skipped > 10
            && cov.scored_c > 100
            && cov.stepped > 100,
        "VOID: the KAT did not exercise every branch: {cov:?}"
    );
}

#[test]
fn cuda_fd_level_matches_cpu_f32_on_tiny_and_cifar_shaped_levels() {
    let test = "cuda_fd_level_matches_cpu_f32_on_tiny_and_cifar_shaped_levels";
    let Some(dev) = search_device(test) else {
        return;
    };
    let mut cov = Coverage::default();
    for (n, depth, tag) in [
        (11_418usize, 28usize, "tiny-like"),
        (3_448, 45, "cifar-like"),
    ] {
        let p = problem(n, depth, 0xFD4, false);
        let lv = level(45, depth, 0xFD4, false);
        kat(tag, &dev, &p, depth, &lv, &mut cov);
    }
    eprintln!("[cuda-fd-kat] {test}: {cov:?}");
    assert!(cov.nodes == 90 && cov.far_moved > 0, "VOID: {cov:?}");
}

#[test]
fn cuda_fd_level_is_launch_and_host_thread_independent() {
    let test = "cuda_fd_level_is_launch_and_host_thread_independent";
    let Some(dev) = search_device(test) else {
        return;
    };
    let depth = 20;
    let count = 7 * 16 + 9;
    let p = problem(777, depth, 31, false);
    let (sides, l0, l1) = level(count, depth, 31, false);
    let digest = |o: &LevelOut| {
        let v = o.hint().into_ingest();
        let mut out = Vec::new();
        for k in 0..v.count() {
            for pt in 0..FD_POINTS {
                out.push(canon(v.value(k, pt)));
                out.push(canon(v.scale(k, pt)));
            }
            for (a, b) in [v.clamped(k), v.far(k), v.next(k)] {
                out.extend(a.iter().chain(b).map(|&x| canon(x)));
            }
        }
        out
    };
    let whole = digest(&run(&dev, &p, &chunk(depth, &sides, &l0, &l1)).0);
    for threads in [1usize, 2, 7] {
        let cpu = ny_bbe::CpuF32Backend::with_threads(threads);
        assert_eq!(
            whole,
            digest(&run(&cpu, &p, &chunk(depth, &sides, &l0, &l1)).0),
            "the device answer is not the CPU answer at {threads} threads"
        );
    }
    // The lane may cut a level anywhere.
    let h = dev.upload_fd(&p, &BIG).expect("upload");
    let mut joined = Vec::new();
    let mut lo = 0usize;
    for cut in [1usize, 17, 40, count] {
        let r = lo * depth..cut * depth;
        let mut out = LevelOut::default();
        let _ = dev
            .fd_level(
                &h,
                &chunk(depth, &sides[r.clone()], &l0[r.clone()], &l1[r]),
                &mut out,
            )
            .expect("fd_level");
        joined.extend(digest(&out));
        lo = cut;
    }
    assert_eq!(whole, joined, "cutting the level changed the device search");
}

/// MUTATION PIN of the KAT itself: the comparator every test above relies on
/// must catch a ONE-ULP change of any value, `S` or proposed multiplier, and
/// must NOT trip on what is not an output (the sign of a zero, a NaN's
/// payload). Device-free (CPU outputs mutated by hand), so it runs — and can
/// fail — on every host, including those where the KATs above are VOID.
#[test]
fn cuda_fd_kat_comparator_catches_a_one_ulp_mutation() {
    use ny_core::search::LevelOutWriter;
    let mut p = problem(300, 6, 5, false);
    // Clearly positive nodes: their clamped / floor points are skipped (NaN).
    p.c0 = 1e5;
    let lv = level(20, 6, 5, false);
    let c = chunk(6, &lv.0, &lv.1, &lv.2);
    let (base, _) = run(&ny_bbe::CpuF32Backend::with_threads(2), &p, &c);
    let mut cov = Coverage::default();
    assert_same("identity", &base, &base.clone(), &lv.1, &mut cov);
    let trips = |mutate: &dyn Fn(&mut LevelOutWriter<'_>) -> usize| -> Option<bool> {
        let mut mutant = base.clone();
        let changed = mutate(&mut mutant.writer());
        if changed == 0 {
            return None;
        }
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut cov = Coverage::default();
            assert_same("mutant", &base, &mutant, &lv.1, &mut cov);
        }))
        .is_err();
        Some(caught)
    };
    let flip_first = |buf: &mut [f32]| -> usize {
        match buf.iter().position(|x| x.is_finite() && *x != 0.0) {
            Some(i) => {
                buf[i] = f32::from_bits(buf[i].to_bits() ^ 1);
                1
            }
            None => 0,
        }
    };
    for (what, got) in [
        ("value", trips(&|w| flip_first(w.value))),
        ("S", trips(&|w| flip_first(w.scale))),
        ("clamped", trips(&|w| flip_first(w.clamped))),
        ("far", trips(&|w| flip_first(w.far))),
        ("next", trips(&|w| flip_first(w.next))),
    ] {
        assert_eq!(
            got,
            Some(true),
            "a one-ulp change of a {what} must fail the KAT"
        );
    }
    let negate_zeros = |buf: &mut [f32]| -> usize {
        let mut k = 0;
        for x in buf.iter_mut().filter(|x| **x == 0.0) {
            *x = -*x;
            k += 1;
        }
        k
    };
    let repayload_nans = |buf: &mut [f32]| -> usize {
        let mut k = 0;
        for x in buf.iter_mut().filter(|x| x.is_nan()) {
            *x = f32::from_bits(0x7FC0_0001);
            k += 1;
        }
        k
    };
    assert_eq!(
        trips(&|w| negate_zeros(w.next) + negate_zeros(w.clamped)),
        Some(false),
        "the sign of a zero is not an output"
    );
    assert_eq!(
        trips(&|w| repayload_nans(w.value)),
        Some(false),
        "a NaN's payload is not an output (and the fixture must skip some points)"
    );
}
