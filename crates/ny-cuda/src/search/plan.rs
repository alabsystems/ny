// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The device search's PURE side: the launch geometry shared with
//! `kernels/fd_search.cu`, the lease accounting (every device buffer the
//! backend allocates, derived from ONE layout that the allocation code also
//! reads), and host mirrors of the kernels' `exp` / `ln` for the init
//! self-test. Nothing here touches the driver, so all of it is tested on
//! hosts without a GPU.

use ny_core::search::{Refusal, FD_LOG_BUCKETS, FD_SEARCH_TOL, FD_SUM_TILE};

/// Nodes per CTA of `fd_level_warm` / items per CTA of `fd_level_points`
/// (the kernel's `FD_NB`).
pub(crate) const FD_NB: usize = 8;
/// Threads per CTA of the level kernels (`FD_THREADS`).
pub(crate) const FD_THREADS: u32 = 256;
/// Threads of the single-CTA `fd_compact` (`FD_COMPACT_THREADS`).
pub(crate) const FD_COMPACT_THREADS: u32 = 1024;
/// Threads per CTA of `fd_level_finish` (one thread per node).
pub(crate) const FD_FINISH_THREADS: u32 = 128;
/// Column tile of the kernels' sums (`FD_TILE`): the shared contract.
pub(crate) const FD_TILE: usize = FD_SUM_TILE;
/// Line-search buckets (`FD_BUCKETS`).
pub(crate) const FD_BUCKETS: usize = FD_LOG_BUCKETS;
/// Per-node vectors of `D` floats in the `VEC` scratch (`FD_NVEC`).
pub(crate) const FD_NVEC: usize = 11;
/// Entries of the constant block `K` (`FD_NK`).
pub(crate) const FD_NK: usize = 5;
/// `EVALS` marker of a refused node (`EVALS_REFUSED`).
pub(crate) const EVALS_REFUSED: u32 = 0x8000_0000;

// The kernel sums in the CPU search's tiles and buckets (the bit identity).
const _: () = assert!(FD_TILE == 256 && FD_BUCKETS == 256);

/// Nodes of one slab (one launch sequence) at most: the lane's own chunk cap
/// (`WIDE_CHUNK_MAX`), and small enough that `node << 2` item ids fit `u32`.
pub(crate) const MAX_SLAB_NODES: usize = 16_384;

/// Work of one slab, in `depth * n` column-row units per node: about a 50 ms
/// launch sequence (ENGINE_PROGRAM §1.4 contracts) on a GB10-class device at
/// the kernels' estimated rate. UNMEASURED — the deadline is polled between
/// slabs, so this bounds the cancellation latency, not correctness.
pub(crate) const SLAB_WORK_BUDGET: u64 = 1 << 30;

/// Device memory never leased on a discrete device (driver, context and
/// other processes' headroom; ENGINE_PROGRAM §1.8).
pub(crate) const DEVICE_RESERVE_BYTES: u64 = 3 << 29;

/// The largest generator count / depth the kernels' `int` indices admit.
pub(crate) const MAX_DIM: usize = 1 << 28;

/// The constants block `K` exactly as the CPU search rounds them
/// (`R::from_f64` of each f64 literal): `[tol, floor_e, lw_min, kb,
/// far_scale]`.
#[must_use]
pub(crate) fn constants() -> [f32; FD_NK] {
    [
        FD_SEARCH_TOL as f32,
        1e-12_f64 as f32,
        1e-30_f64 as f32,
        FD_LOG_BUCKETS as f64 as f32,
        1e6_f64 as f32,
    ]
}

/// `gamma_k` of binary32 (`k u / (1 - k u)`, `u = 2^-24`).
fn gamma_f32(k: usize) -> f64 {
    let ku = k as f64 * f64::from(f32::EPSILON) / 2.0;
    if ku < 1.0 {
        ku / (1.0 - ku)
    } else {
        f64::INFINITY
    }
}

/// The far-first skip threshold at depth `d`, rounded as the CPU search
/// rounds it (`R::from_f64(4 gamma_f32(d))`).
#[must_use]
pub(crate) fn rel_threshold(depth: usize) -> f32 {
    (4.0 * gamma_f32(depth)) as f32
}

// ---------------------------------------------------------------------------
//  The device buffers (one layout, read by the planner AND the allocator)
// ---------------------------------------------------------------------------

/// Element type of a device buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Elem {
    F32,
    I32,
    U32,
    U8,
}

impl Elem {
    const fn bytes(self) -> u64 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::U8 => 1,
        }
    }
}

/// Every buffer of one upload (`n` generators, `depth` rows): `(name, elem,
/// len)`. The allocator allocates exactly these (`max(len, 1)` elements).
#[must_use]
pub(crate) fn upload_buffers(n: usize, depth: usize) -> [(&'static str, Elem, usize); 6] {
    [
        ("a", Elem::F32, depth.saturating_mul(n)),
        ("own", Elem::I32, n),
        ("d", Elem::F32, n),
        ("pairs", Elem::F32, depth.saturating_mul(8)),
        ("row_abs", Elem::F32, depth),
        ("k", Elem::F32, FD_NK),
    ]
}

/// Every buffer of one slab: `nodes` nodes of per-node state and `nd`
/// node-rows of per-(node, split) state (a launch of `c` nodes at depth `d`
/// needs `c <= nodes` and `c * d <= nd`; the kernels lay a launch's vectors
/// out with the launch's own node count as the stride).
#[must_use]
pub(crate) fn slab_buffers(nodes: usize, nd: usize) -> [(&'static str, Elem, usize); 16] {
    [
        ("sides", Elem::U8, nd),
        ("l0", Elem::F32, nd),
        ("l1", Elem::F32, nd),
        ("vec", Elem::F32, nd.saturating_mul(FD_NVEC)),
        ("lm", Elem::F32, nd.saturating_mul(2)),
        ("val", Elem::F32, nodes.saturating_mul(4)),
        ("abs", Elem::F32, nodes.saturating_mul(4)),
        ("flags", Elem::U32, nodes),
        ("slope", Elem::F32, nodes),
        ("eta", Elem::F32, nodes),
        ("evals", Elem::U32, nodes),
        ("items", Elem::U32, nodes.saturating_mul(2)),
        ("item_count", Elem::U32, 1),
        ("out_clamped", Elem::F32, nd.saturating_mul(2)),
        ("out_far", Elem::F32, nd.saturating_mul(2)),
        ("out_next", Elem::F32, nd.saturating_mul(2)),
    ]
}

/// The elements of each slab buffer the kernels INDEX for one launch of
/// `count` nodes at depth `d` (`kernels/fd_search.cu`: per-node vectors are
/// laid out with the launch's node count as their stride; the points kernel's
/// item rows run up to `2 count`). The device checks every launch against the
/// buffers it actually allocated, so no kernel index can pass an allocation.
#[must_use]
pub(crate) fn launch_needs(count: usize, d: usize) -> [(&'static str, usize); 16] {
    let cd = count.saturating_mul(d);
    [
        ("sides", cd),
        ("l0", cd),
        ("l1", cd),
        ("vec", cd.saturating_mul(FD_NVEC)),
        ("lm", cd.saturating_mul(2)),
        ("val", count.saturating_mul(4)),
        ("abs", count.saturating_mul(4)),
        ("flags", count),
        ("slope", count),
        ("eta", count),
        ("evals", count),
        ("items", count.saturating_mul(2)),
        ("item_count", 1),
        ("out_clamped", cd.saturating_mul(2)),
        ("out_far", cd.saturating_mul(2)),
        ("out_next", cd.saturating_mul(2)),
    ]
}

fn bytes_of(buffers: &[(&'static str, Elem, usize)]) -> u64 {
    buffers
        .iter()
        .map(|&(_, e, len)| (len.max(1) as u64).saturating_mul(e.bytes()))
        .fold(0u64, u64::saturating_add)
}

/// Device bytes of one upload.
#[must_use]
pub(crate) fn upload_bytes(n: usize, depth: usize) -> u64 {
    bytes_of(&upload_buffers(n, depth))
}

/// Device bytes of one slab of `nodes` nodes and `nd` node-rows.
#[must_use]
pub(crate) fn slab_bytes(nodes: usize, nd: usize) -> u64 {
    bytes_of(&slab_buffers(nodes, nd))
}

/// The memory a discrete device may lease (`min(free - reserve, 0.85
/// total)`, ENGINE_PROGRAM §1.8); a unified device's memory is the host's,
/// whose headroom the lane's lease already is, so the same clamp only
/// tightens it.
#[must_use]
pub(crate) fn device_cap(free: u64, total: u64) -> u64 {
    free.saturating_sub(DEVICE_RESERVE_BYTES)
        .min(total / 20 * 17)
}

/// One upload's admission: the bytes it holds and its slab capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Bytes of the uploaded problem.
    pub(crate) upload: u64,
    /// Nodes a slab holds.
    pub(crate) slab_nodes: usize,
    /// Node-rows a slab holds.
    pub(crate) slab_nd: usize,
    /// Bytes of the slab buffers.
    pub(crate) slab: u64,
}

impl Plan {
    /// Every device byte the upload holds.
    #[must_use]
    pub(crate) fn total(&self) -> u64 {
        self.upload.saturating_add(self.slab)
    }

    /// A slab of exactly `nodes` nodes and `nd` node-rows over `(n, depth)`.
    #[must_use]
    pub(crate) fn exact(n: usize, depth: usize, nodes: usize, nd: usize) -> Self {
        Self {
            upload: upload_bytes(n, depth),
            slab_nodes: nodes,
            slab_nd: nd,
            slab: slab_bytes(nodes, nd),
        }
    }
}

/// Nodes of one launch at `depth` over `n` generators within the work budget
/// (at least one CTA, at most [`MAX_SLAB_NODES`]), before the slab caps it.
pub(crate) fn budget_nodes(n: usize, depth: usize) -> usize {
    let per_node = (depth.max(1) as u64).saturating_mul(n.max(1) as u64);
    usize::try_from(SLAB_WORK_BUDGET / per_node)
        .unwrap_or(usize::MAX)
        .clamp(FD_NB, MAX_SLAB_NODES)
}

/// Admit an upload of `(n, depth)` under `cap` bytes. The slab is sized to
/// what the work budget can ask of it at ANY chunk depth `1..=depth` (so a
/// deep, wide problem does not reserve `MAX_SLAB_NODES x depth` rows it would
/// never fill), shrunk to fit the cap if it must; the problem plus a slab of
/// [`FD_NB`] nodes at full depth is the minimum. A refusal is
/// [`Refusal::Memory`] with that minimum need.
pub(crate) fn plan(n: usize, depth: usize, cap: u64) -> Result<Plan, Refusal> {
    let d = depth.max(1);
    let upload = upload_bytes(n, depth);
    let min_need = upload.saturating_add(slab_bytes(FD_NB, FD_NB * d));
    if min_need > cap {
        return Err(Refusal::Memory {
            need: min_need,
            cap,
        });
    }
    let want_nodes = budget_nodes(n, 1);
    let want_nd = (1..=d)
        .map(|dc| dc.saturating_mul(budget_nodes(n, dc)))
        .max()
        .unwrap_or(FD_NB * d);
    let fits = |k: usize| {
        let nd = want_nd.min(k.saturating_mul(d)).max(FD_NB * d);
        upload.saturating_add(slab_bytes(k, nd)) <= cap
    };
    // The largest node capacity that fits (monotone in `k`).
    let (mut lo, mut hi) = (FD_NB, want_nodes.max(FD_NB));
    if fits(hi) {
        lo = hi;
    }
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let nd = want_nd.min(lo.saturating_mul(d)).max(FD_NB * d);
    Ok(Plan {
        upload,
        slab_nodes: lo,
        slab_nd: nd,
        slab: slab_bytes(lo, nd),
    })
}

/// Nodes of one launch at `depth` over `n` generators on a slab of `plan`:
/// within the work budget and the slab's node and node-row capacity (at
/// least one).
#[must_use]
pub(crate) fn slab_nodes(n: usize, depth: usize, plan: &Plan) -> usize {
    budget_nodes(n, depth)
        .min(plan.slab_nodes)
        .min(plan.slab_nd / depth.max(1))
        .max(1)
}

// ---------------------------------------------------------------------------
//  Host mirrors of the kernels' math (the init self-test's oracle)
// ---------------------------------------------------------------------------

/// Host copy of the kernels' `fast_ln` (= ny-bbe `fd::fast_ln_f32`).
#[must_use]
pub(crate) fn host_fast_ln(x: f32) -> f32 {
    if x == f32::INFINITY {
        return x;
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > std::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let series = 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * (0.2 + s2 * (1.0 / 7.0))));
    e as f32 * std::f32::consts::LN_2 + series
}

/// Host copy of the kernels' `fast_exp` (= ny-bbe `fd::fast_exp_f32`).
#[must_use]
pub(crate) fn host_fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B);
    const LN2_HI: f32 = f32::from_bits(0x3F31_8000);
    const LN2_LO: f32 = f32::from_bits(0xB95E_8083);
    const HI: f32 = f32::from_bits(0x42B1_7217);
    const LO: f32 = f32::from_bits(0xC2AE_AC4F);
    const C3: f32 = f32::from_bits(0x3E2A_AAAB);
    const C4: f32 = f32::from_bits(0x3D2A_AAAB);
    const C5: f32 = f32::from_bits(0x3C08_8889);
    const C6: f32 = f32::from_bits(0x3AB6_0B61);
    if x.is_nan() {
        return x;
    }
    if x > HI {
        return f32::INFINITY;
    }
    if x < LO {
        return 0.0;
    }
    let nf = (x * LOG2E + 0.5).floor();
    let r = (x - nf * LN2_HI) - nf * LN2_LO;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (C3 + r * (C4 + r * (C5 + r * C6)))));
    let n = nf as i32;
    let h = n / 2;
    let pow2 = |k: i32| f32::from_bits(((k + 127) as u32) << 23);
    p * pow2(h) * pow2(n - h)
}

/// The self-test's math inputs: the documented edges and a sweep of both
/// functions' domains.
#[must_use]
pub(crate) fn math_kat_inputs() -> Vec<f32> {
    let mut xs = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        f32::from_bits(0x42B1_7217),
        f32::from_bits(0xC2AE_AC4F),
        89.0,
        -88.0,
        1e-12,
        1e-30,
        3.0e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
    ];
    for k in 0..128u16 {
        xs.push(-87.0 + f32::from(k) * 1.37);
    }
    let mut y = 1e-12f32;
    for _ in 0..50 {
        xs.push(y);
        xs.push(y * 1.414_3);
        y *= 7.1;
    }
    xs
}

/// Same bits, or both NaN (a NaN's payload is not an output anyone reads).
#[must_use]
pub(crate) fn same_bits(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

// ---------------------------------------------------------------------------
//  The init self-test's level fixture (exact: every sum is dyadic)
// ---------------------------------------------------------------------------

/// splitmix64 step.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `k / den` with `k` uniform in `lo..=hi`.
fn dyadic(state: &mut u64, lo: i64, hi: i64, den: f32) -> f32 {
    let span = (hi - lo + 1) as u64;
    let k = lo + (mix(state) % span) as i64;
    k as f32 / den
}

/// A dyadic frozen-dual search copy whose every warm sum is exact in f32
/// (entries `k/16`, multipliers `k/8`, `n * depth` small): the warm value and
/// `S` then have ONE correct f32 answer, independent of summation order.
#[must_use]
pub(crate) fn kat_problem(n: usize, depth: usize, seed: u64) -> ny_core::search::FdSearchProblem {
    let mut s = seed ^ 0x00C0_DA5E;
    let depth = depth.min(n);
    let mut cols: Vec<u32> = (0..n as u32).collect();
    for i in 0..depth {
        let j = i + (mix(&mut s) % (n - i) as u64) as usize;
        cols.swap(i, j);
    }
    let a: Vec<f32> = (0..depth * n)
        .map(|_| {
            if mix(&mut s).is_multiple_of(5) {
                0.0
            } else {
                dyadic(&mut s, -16, 16, 16.0)
            }
        })
        .collect();
    let row_abs: Vec<f32> = (0..depth)
        .map(|i| a[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum())
        .collect();
    let mut pair =
        |lo: i64, hi: i64, den: f32| [dyadic(&mut s, lo, hi, den), dyadic(&mut s, lo, hi, den)];
    let rho: Vec<[f32; 2]> = (0..depth).map(|_| pair(-16, 16, 8.0)).collect();
    let c0s: Vec<[f32; 2]> = (0..depth).map(|_| pair(-16, 0, 32.0)).collect();
    let b0: Vec<[f32; 2]> = (0..depth).map(|_| pair(0, 48, 16.0)).collect();
    let b1: Vec<[f32; 2]> = (0..depth).map(|_| pair(0, 48, 16.0)).collect();
    let d: Vec<f32> = (0..n).map(|_| dyadic(&mut s, -16, 16, 16.0)).collect();
    let abs_d: f32 = d.iter().map(|v| v.abs()).sum();
    ny_core::search::FdSearchProblem {
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
        c0: (0.5 * abs_d).floor(),
        abs_d,
    }
}

/// A dyadic level of `count` nodes of `depth` for [`kat_problem`].
#[must_use]
pub(crate) fn kat_level(count: usize, depth: usize, seed: u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut s = seed ^ 0x1E7E_1000;
    let mut sides = Vec::with_capacity(count * depth);
    let mut l0 = Vec::with_capacity(count * depth);
    let mut l1 = Vec::with_capacity(count * depth);
    for _ in 0..count * depth {
        sides.push(u8::from(mix(&mut s) % 2 == 1));
        let mut m = || {
            if mix(&mut s).is_multiple_of(4) {
                0.0
            } else {
                dyadic(&mut s, 0, 16, 8.0)
            }
        };
        l0.push(m());
        l1.push(m());
    }
    (sides, l0, l1)
}

/// The exact warm value and `S` of one node of a dyadic problem (sequential
/// f64 sums, exact here, so the unique f32 answer).
#[must_use]
pub(crate) fn exact_warm(
    p: &ny_core::search::FdSearchProblem,
    sides: &[u8],
    l0: &[f32],
    l1: &[f32],
) -> (f32, f32) {
    let (n, d) = (p.n, sides.len());
    let mut rc: Vec<f64> = p.d.iter().map(|&v| f64::from(v)).collect();
    for &e in &p.enc[..d] {
        rc[e as usize] = 0.0;
    }
    let mut c0path = f64::from(p.c0);
    let mut c0abs = c0path.abs();
    let (mut lag, mut lagabs, mut tsum) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..d {
        let a = f64::from(l0[i].max(0.0));
        let b = f64::from(l1[i].max(0.0));
        let si = usize::from(sides[i]);
        let sign = if si == 0 { 1.0 } else { -1.0 };
        let rho = f64::from(p.rho[i][si]);
        let mult = rho + sign * (a - b);
        for (j, r) in rc.iter_mut().enumerate() {
            *r += mult * f64::from(p.a[i * n + j]);
        }
        let cs = f64::from(p.c0s[i][si]);
        c0path += cs;
        c0abs += cs.abs();
        let (b0, b1) = (f64::from(p.b0[i][si]), f64::from(p.b1[i][si]));
        lag += a * b0 + b * b1;
        lagabs += a * b0.abs() + b * b1.abs();
        tsum += (rho.abs() + a + b) * f64::from(p.row_abs[i]);
    }
    let sumabs: f64 = rc.iter().map(|v| v.abs()).sum();
    let value = c0path - sumabs - lag;
    let big_s = c0abs + sumabs + f64::from(p.abs_d) + tsum + lagabs;
    (value as f32, big_s as f32)
}
