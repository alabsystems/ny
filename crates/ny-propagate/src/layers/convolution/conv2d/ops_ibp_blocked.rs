// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Blocked-pollable twin of the finite-deadline CERTIFIED f64 Conv2d IBP
//! forward (#pollable-conv-ibp).
//!
//! # The problem this removes
//!
//! `graph_alpha/bounds/ibp.rs` dispatches every Conv2d node of a
//! deadline-carrying graph IBP collection to
//! `Conv2dLayer::propagate_ibp_sound_with_engine_and_deadline`, whose finite
//! arm is `conv2d_ibp_forward_grouped_certified_f64_with_deadline` — a serial
//! per-MAC scalar contraction. That kernel is CORRECT and its enclosure is the
//! tight one (#cgan-conv-ibp-magnitude-floor replaced a magnitude-scaled f32
//! widening with it, and the cgan tree only closes because of that). What it
//! is not is fast: it pays, per MAC,
//!
//! * three `IxDyn` dynamic-stride index computations with bounds checks
//!   (`kernel[[oc, ic_local, ki, kj]]`, `input_lower[[ic, ih, iw]]`,
//!   `input_upper[[ic, ih, iw]]`),
//! * a full `conv2d_tap_source` resolution — four `checked_*` operations and a
//!   `filter` — even though tap validity depends only on `(oh, ow, ki, kj)` and
//!   is therefore re-derived `out_c` times,
//! * a poll counter increment and compare,
//!
//! all on one core, with a data-dependent branch in the innermost statement
//! that blocks vectorisation. Measured consequence: `attack_point_gradient` on
//! CIFAR100_resnet_large at **8.5 s/step** against **~93 ms/step** for the same
//! call with no layer deadline (`graph_crown/point_vjp_batched_resnet.rs:9-12`)
//! — a ~91x tax paid for INTRA-node cancellation.
//!
//! # What this module does instead
//!
//! The same certificate, the same taps, the same fold order per output — with
//! the interpretation overhead hoisted out and the deadline observed between
//! MAC-budgeted OUTPUT BLOCKS. Concretely:
//!
//! 1. **Tap tables.** `conv2d_tap_axis` is separable, so the live taps of a
//!    window are `row_taps[oh] x col_taps[ow]`. Both tables are built once per
//!    call (`out_h·kh + out_w·kw` entries) instead of `out_c·out_h·out_w·kh·kw`
//!    times.
//! 2. **Kernel transposed to `[g][ic_local][ki][kj][oc_local]`** once per call,
//!    so the innermost loop reads a CONTIGUOUS `out_c_per_group`-long weight
//!    row — the `wt` layout `margin_row/net.rs` already uses for exactly this
//!    reason.
//! 3. **Accumulate all output channels of a group at once** for a fixed
//!    `(group, output position)`. The input pair `(l, u)` and the abs-magnitude
//!    `m` are loop-invariant across `oc_local`, so the inner loop is a
//!    contiguous select+multiply+add over three accumulator arrays — the shape
//!    an autovectoriser can take.
//! 4. **Poll between blocks**, where a block is `CONV_IBP_BLOCK_MACS` worth of
//!    work, not a fixed count — the `#pollable-faer` / `DEADLINE_AW_FAER_CHUNK_MACS`
//!    rule, so the quantum adapts to layer shape.
//! 5. **Optionally fan the block out over Rayon**, which is bit-safe because
//!    every output element is an independent fold (the
//!    `conv_backward_blocked` argument in `margin_row/net.rs`).
//!
//! # Bit-identity (this is the bar, and it is met by construction)
//!
//! For a fixed output `(oc, oh, ow)` the scalar kernel folds
//!
//! ```text
//! for ic_local in 0..in_c_per_group { for ki in 0..kh { for kj in 0..kw {
//!     if the tap is live { acc_{lo,hi,abs} += ... }
//! } } }
//! ```
//!
//! and this kernel folds `for ic_local { for (ki,·) in row_taps { for (kj,·) in
//! col_taps { ... } } }`. `row_taps[oh]` is ascending in `ki` and `col_taps[ow]`
//! ascending in `kj`, and they contain exactly the indices for which
//! `conv2d_tap_axis` returned `Some` — the same predicate, the same function,
//! the same order, the same skips. Each accumulator therefore receives the
//! IDENTICAL sequence of f64 additions. Interleaving the `oc_local` lanes does
//! not reorder any single lane's sequence, and neither does block or thread
//! assignment: a block boundary always falls BETWEEN whole output elements, so
//! it cannot split a reduction. `m = l.abs().max(u.abs())` is hoisted out of
//! the `oc_local` loop, which is a common-subexpression move, not a
//! reassociation. Bias, `err` ([`certified_err_term`]) and the directed casts
//! come from the shared helpers in `ops_ibp_fwd.rs`, so the certificate cannot
//! drift.
//!
//! Consequence: this route publishes BYTE-IDENTICAL bounds to the scalar
//! oracle. It is a throughput change only — `MoatRisk::None` — and that is why
//! it may ship default-ON with `NY_CONV_IBP_BLOCKED=0` as the kill switch.
//! Pinned by `conv2d_blocked_certified_f64_bit_identical_to_scalar_route`.
//!
//! # Why NOT im2col + GEMM
//!
//! The obvious alternative — restore the fast route's im2col + `gemm_f32`
//! under a chunked cancellation contract — was rejected on evidence, twice
//! over:
//!
//! * The `deadline: None` twin of THIS kernel is not an f32 GEMM of the same
//!   quantity. It is a round-to-nearest f32 GEMM plus a
//!   `γ_{K+2}^{f32}·S + 2u·|y|` widening whose charge scales with activation
//!   MAGNITUDE. Routing the finite-deadline arm back onto it is precisely the
//!   `6f49a660` regression that `#cgan-conv-ibp-magnitude-floor` diagnosed and
//!   reverted (cGAN_imgSz32_nCh_3_prop_2: unsat@640s before, timeout@1800s
//!   after). An f32 engine cannot carry the f64 certificate.
//! * An f64 im2col + `mat_mul_f64` WOULD keep the certificate (γ_n·S is
//!   summation-order independent, the `aw_f64_with_abssum_chunked_poll`
//!   argument), but it needs five GEMMs per group (`l@W+`, `u@W-`, `u@W+`,
//!   `l@W-`, `max(|l|,|u|)@|W|`), materialises three f64 im2col tiles, and is
//!   NOT bit-identical to what ships today — so it could only land behind a
//!   measured verdict A/B. Given a bit-identical restructuring reaches the same
//!   arithmetic intensity class, it is strictly worse here.
//!
//! If a future measurement shows this kernel still leaves a large gap to the
//! `deadline: None` route, the f64-GEMM variant is the next step and the
//! certificate argument for it is already written above.

use ndarray::{Array1, ArrayD, ArrayViewD};
use ny_core::{checked_shape_product, NyError, Result};
use rayon::prelude::*;
use std::sync::OnceLock;
use std::time::Instant;

use super::ops_ibp_fwd::{
    certified_cast_lower_f32, certified_cast_upper_f32, certified_conv_deadline_error,
    certified_err_term, certified_f64_gamma_and_inflate, conv2d_deadline_geometry, conv2d_tap_axis,
    Conv2dDeadlineGeometry, ConvIbpForward,
};

/// Multiply/accumulate work admitted between finite-deadline polls on the
/// blocked route.
///
/// A MAC BUDGET, not an output count, so the quantum adapts to layer shape —
/// the `POLLABLE_FAER_BLOCK_MACS` / `DEADLINE_AW_FAER_CHUNK_MACS` rule. Sized
/// from the two throughputs that bracket this kernel:
///
/// * the scalar oracle it replaces sustains ~84M MAC/s (718M serial per-MAC
///   iterations in 8.5 s, `graph_alpha/bounds/ibp.rs`), so 2^20 MACs is ~12 ms
///   there — the WORST case, and only reachable if the hoisting buys nothing;
/// * a contiguous, tap-hoisted f64 fold is conservatively ~0.3 GMAC/s
///   sequential and several GMAC/s across a Rayon block, putting one block at
///   ~3.5 ms and ~0.3 ms respectively.
///
/// That band sits inside the tolerance `#pollable-faer` already accepted for
/// this class of seam (2.75 ms on the driver thread, ~6 ms in the `Par::Seq`
/// rayon-worker regime, `crown_batched.rs`), and it is three orders of
/// magnitude finer than the per-NODE granularity the graph collection loop
/// falls back to when a layer deadline is withheld.
///
/// The floor is one output element's `out_c_per_group·K` fold, which must stay
/// whole (a reduction may not be split across polls without changing its
/// summation order) — the same "a single row's `k·p` must stay whole"
/// concession `aw_f64_with_abssum_chunked_poll` makes.
pub(crate) const CONV_IBP_BLOCK_MACS: usize = 1 << 20;

/// Ceiling on the per-block staging buffer of the PARALLEL arm, in f32
/// elements per side. Only relevant for tiny-`K` layers (1x1 convs), where the
/// MAC budget alone would admit a very large output block; this box's 121 GiB
/// is SHARED between CPU and GPU (#gb10-unified-memory-oom-trap), so every
/// scratch allocation on a hot path gets an explicit cap.
const CONV_IBP_PARALLEL_BLOCK_ELEMS: usize = 1 << 18;

/// Which finite-deadline certified Conv2d IBP kernel runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvIbpBlockedRoute {
    /// The historical per-MAC scalar contraction — the bit-identity ORACLE,
    /// selected by the kill switch.
    Scalar,
    /// Blocked kernel, single-threaded.
    BlockedSequential,
    /// Blocked kernel, Rayon-parallel within each block. The shipped default.
    BlockedParallel,
}

/// The lever's arming rule, as a pure unit-tested predicate (the
/// `blocked_backward_enabled_from_env` idiom — the predicate IS the spec, so it
/// stays in the production path rather than being restated in a test).
///
/// * `0`, `false`, `off`, `no` (any case, surrounding whitespace trimmed) —
///   [`ConvIbpBlockedRoute::Scalar`], byte-identical to the history.
/// * `seq` / `sequential` — [`ConvIbpBlockedRoute::BlockedSequential`], the
///   single-threaded blocked arm, for isolating "blocking" from "threads" in a
///   measurement.
/// * absent, empty, or anything else — [`ConvIbpBlockedRoute::BlockedParallel`].
pub(crate) fn conv_ibp_blocked_route_from_token(raw: Option<&str>) -> ConvIbpBlockedRoute {
    let Some(value) = raw.map(str::trim) else {
        return ConvIbpBlockedRoute::BlockedParallel;
    };
    if value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("no")
    {
        return ConvIbpBlockedRoute::Scalar;
    }
    if value.eq_ignore_ascii_case("seq") || value.eq_ignore_ascii_case("sequential") {
        return ConvIbpBlockedRoute::BlockedSequential;
    }
    ConvIbpBlockedRoute::BlockedParallel
}

/// Route selector, latched once.
///
/// The central preset gate supplies the winning raw/canonical token, while the
/// decision stays this site's own unit-tested predicate. The latch is
/// process-wide, which is the accepted Phase-2 debt for every gate of this
/// shape; tests exercise the predicate directly and the kernels directly,
/// never through this latch.
fn conv_ibp_blocked_route() -> ConvIbpBlockedRoute {
    static ROUTE: OnceLock<ConvIbpBlockedRoute> = OnceLock::new();
    *ROUTE.get_or_init(|| {
        let token = crate::preset_gates::conv_ibp_blocked_token();
        conv_ibp_blocked_route_from_token(token.as_deref())
    })
}

/// The route the shipped finite-deadline certified Conv2d IBP forward takes.
///
/// R9: this prints ONE line the first time the route is actually taken. A lever
/// with no runtime observable cannot produce a non-vacuous measurement — a conv
/// A/B was run and voided for exactly that reason (the arms were
/// indistinguishable in the logs, and the lane turned out not to reach here at
/// all). `eprintln!` and not `info!`: this binary swallows `info!` on the
/// scored route, which is how the cgan dark tail stayed invisible for weeks.
pub(crate) fn certified_conv_ibp_route() -> ConvIbpBlockedRoute {
    let route = conv_ibp_blocked_route();
    static ANNOUNCED: OnceLock<()> = OnceLock::new();
    ANNOUNCED.get_or_init(|| {
        let arm = match route {
            ConvIbpBlockedRoute::Scalar => "scalar-oracle",
            ConvIbpBlockedRoute::BlockedSequential => "blocked-seq",
            ConvIbpBlockedRoute::BlockedParallel => "blocked-parallel",
        };
        eprintln!(
            "[conv-ibp-blocked] route={arm} block_macs={CONV_IBP_BLOCK_MACS} \
             (finite-deadline certified Conv2d IBP forward; absence of this line \
             means the lane never reached this kernel)"
        );
    });
    route
}

/// Everything one work unit — one `(group, output position)` pair — needs.
/// Shared immutably across Rayon workers; every field is `Sync`.
struct BlockedCtx<'a> {
    lower: &'a [f32],
    upper: &'a [f32],
    /// Kernel transposed to `[g][ic_local][ki][kj][oc_local]`.
    weights: &'a [f32],
    /// `row_taps[oh]` = live `(ki, ih)` pairs, ascending in `ki`.
    row_taps: &'a [Vec<(usize, usize)>],
    /// `col_taps[ow]` = live `(kj, iw)` pairs, ascending in `kj`.
    col_taps: &'a [Vec<(usize, usize)>],
    bias: Option<&'a Array1<f32>>,
    in_c_per_group: usize,
    out_c_per_group: usize,
    kh: usize,
    kw: usize,
    in_w: usize,
    in_spatial: usize,
    out_w: usize,
    gamma: f64,
    abs_inflate: f64,
}

/// Fold ONE `(group, output position)` unit over all `out_c_per_group` output
/// channels and publish their certified f32 endpoints.
///
/// `acc_*` are caller-owned scratch of length `>= out_c_per_group` (reused
/// across units so the hot path allocates nothing); `out_*` receive exactly
/// `out_c_per_group` published endpoints in ascending `oc_local`.
///
/// The per-lane addition sequence is the scalar oracle's, verbatim — see the
/// module header for the argument.
#[allow(clippy::too_many_arguments)]
#[inline]
fn certified_unit(
    ctx: &BlockedCtx<'_>,
    group: usize,
    position: usize,
    acc_lo: &mut [f64],
    acc_hi: &mut [f64],
    acc_abs: &mut [f64],
    out_lo: &mut [f32],
    out_hi: &mut [f32],
) {
    let ocpg = ctx.out_c_per_group;
    let acc_lo = &mut acc_lo[..ocpg];
    let acc_hi = &mut acc_hi[..ocpg];
    let acc_abs = &mut acc_abs[..ocpg];
    acc_lo.fill(0.0);
    acc_hi.fill(0.0);
    acc_abs.fill(0.0);

    let oh = position / ctx.out_w;
    let ow = position % ctx.out_w;
    let rows = &ctx.row_taps[oh];
    let cols = &ctx.col_taps[ow];

    for ic_local in 0..ctx.in_c_per_group {
        let channel = group * ctx.in_c_per_group + ic_local;
        let x_base = channel * ctx.in_spatial;
        // weights index = (((g·icpg + ic_local)·kh + ki)·kw + kj)·ocpg
        let w_channel = (group * ctx.in_c_per_group + ic_local) * ctx.kh * ctx.kw;
        for &(ki, ih) in rows {
            let x_row = x_base + ih * ctx.in_w;
            let w_row = (w_channel + ki * ctx.kw) * ocpg;
            for &(kj, iw) in cols {
                // Loop-invariant across `oc_local`: hoisting these is common
                // subexpression elimination, not reassociation.
                let l = f64::from(ctx.lower[x_row + iw]);
                let u = f64::from(ctx.upper[x_row + iw]);
                let magnitude = l.abs().max(u.abs());
                let w_base = w_row + kj * ocpg;
                let weights = &ctx.weights[w_base..w_base + ocpg];
                for (((lo, hi), abs), &weight32) in acc_lo
                    .iter_mut()
                    .zip(acc_hi.iter_mut())
                    .zip(acc_abs.iter_mut())
                    .zip(weights.iter())
                {
                    let weight = f64::from(weight32);
                    // `w >= 0 ? (l, u) : (u, l)` — the oracle's branch written
                    // as a select so it vectorises; the products and the two
                    // `+=` are bit-for-bit the oracle's.
                    let (lo_operand, hi_operand) = if weight >= 0.0 { (l, u) } else { (u, l) };
                    *lo += weight * lo_operand;
                    *hi += weight * hi_operand;
                    *abs += weight.abs() * magnitude;
                }
            }
        }
    }

    for oc_local in 0..ocpg {
        let mut lo = acc_lo[oc_local];
        let mut hi = acc_hi[oc_local];
        let mut abs = acc_abs[oc_local];
        if let Some(bias) = ctx.bias {
            let value = f64::from(bias[group * ocpg + oc_local]);
            lo += value;
            hi += value;
            abs += value.abs();
        }
        let err = certified_err_term(abs, ctx.gamma, ctx.abs_inflate);
        if err.is_finite() {
            out_lo[oc_local] = certified_cast_lower_f32(lo - err);
            out_hi[oc_local] = certified_cast_upper_f32(hi + err);
        } else {
            // γ saturation, NaN taps, or infinite abs-sum: publish the
            // universal interval (widening-only; never a tight lie).
            out_lo[oc_local] = f32::NEG_INFINITY;
            out_hi[oc_local] = f32::INFINITY;
        }
    }
}

/// Blocked-pollable finite-deadline CERTIFIED grouped Conv2d interval forward.
///
/// Drop-in twin of
/// [`super::ops_ibp_fwd::conv2d_ibp_forward_grouped_certified_f64_with_deadline`]:
/// same arguments, same validation (through the shared
/// [`conv2d_deadline_geometry`]), same published values bit-for-bit, and the
/// SAME typed refusal ([`certified_conv_deadline_error`]) on expiry with no
/// partial result — so every caller's fallback contract is unchanged.
///
/// `block_macs` and `parallel` are explicit so tests can force multi-block
/// dispatch and pin the parallel/sequential parity without touching the
/// process environment; production passes [`CONV_IBP_BLOCK_MACS`] and the
/// latched route.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_ibp_forward_grouped_certified_f64_blocked_with_deadline(
    input_lower: ArrayViewD<'_, f32>,
    input_upper: ArrayViewD<'_, f32>,
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    deadline: Instant,
    block_macs: usize,
    parallel: bool,
) -> Result<ConvIbpForward> {
    let deadline_error = certified_conv_deadline_error;
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    let geometry = conv2d_deadline_geometry(
        &input_lower,
        &input_upper,
        kernel,
        stride,
        padding,
        dilation,
        groups,
    )?;
    let Conv2dDeadlineGeometry {
        in_h,
        in_w,
        out_c,
        in_c_per_group,
        out_c_per_group,
        kh,
        kw,
        out_h,
        out_w,
        spatial,
        out_size,
    } = geometry;
    if let Some(values) = bias {
        if values.len() != out_c {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_c],
                got: vec![values.len()],
            });
        }
    }
    let macs = in_c_per_group.saturating_mul(kh).saturating_mul(kw);
    let (gamma, abs_inflate) = certified_f64_gamma_and_inflate(macs);
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }

    // Contiguous input views. `as_standard_layout` copies only for a
    // non-standard-layout caller, exactly as the no-deadline im2col route does.
    let lower_std = input_lower.as_standard_layout();
    let upper_std = input_upper.as_standard_layout();
    let lower_data = lower_std.as_slice().ok_or_else(|| {
        NyError::InternalError("conv2d_ibp_forward: input_lower not contiguous".to_string())
    })?;
    let upper_data = upper_std.as_slice().ok_or_else(|| {
        NyError::InternalError("conv2d_ibp_forward: input_upper not contiguous".to_string())
    })?;
    let in_spatial = in_h.checked_mul(in_w).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: input spatial size overflow".to_string())
    })?;
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }

    // Kernel transposed to [g][ic_local][ki][kj][oc_local] so the innermost
    // loop reads a contiguous out_c_per_group weight row. Same element count
    // as `kernel`; the copy is exact.
    let weights_len = checked_shape_product(&[out_c, in_c_per_group, kh, kw]).ok_or_else(|| {
        NyError::InvalidSpec("conv2d_ibp_forward: transposed kernel size overflow".to_string())
    })?;
    let mut weights = vec![0.0f32; weights_len];
    let mut copied = 0usize;
    for group in 0..groups {
        let oc_base = group * out_c_per_group;
        for ic_local in 0..in_c_per_group {
            let channel = (group * in_c_per_group + ic_local) * kh * kw;
            for ki in 0..kh {
                for kj in 0..kw {
                    let base = (channel + ki * kw + kj) * out_c_per_group;
                    for oc_local in 0..out_c_per_group {
                        weights[base + oc_local] = kernel[[oc_base + oc_local, ic_local, ki, kj]];
                    }
                    copied = copied.saturating_add(out_c_per_group);
                    if copied >= block_macs.max(1) {
                        copied = 0;
                        if Instant::now() >= deadline {
                            return Err(deadline_error());
                        }
                    }
                }
            }
        }
    }
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }

    // Separable tap tables. `conv2d_tap_axis` is the SAME predicate the scalar
    // oracle applies per MAC, so these enumerate exactly its live taps, in its
    // ascending order.
    let mut row_taps: Vec<Vec<(usize, usize)>> = Vec::with_capacity(out_h);
    for oh in 0..out_h {
        let mut taps = Vec::with_capacity(kh);
        for ki in 0..kh {
            if let Some(ih) = conv2d_tap_axis(oh, ki, stride.0, dilation.0, padding.0, in_h) {
                taps.push((ki, ih));
            }
        }
        row_taps.push(taps);
    }
    let mut col_taps: Vec<Vec<(usize, usize)>> = Vec::with_capacity(out_w);
    for ow in 0..out_w {
        let mut taps = Vec::with_capacity(kw);
        for kj in 0..kw {
            if let Some(iw) = conv2d_tap_axis(ow, kj, stride.1, dilation.1, padding.1, in_w) {
                taps.push((kj, iw));
            }
        }
        col_taps.push(taps);
    }
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }

    let mut lower_flat = vec![0.0f32; out_size];
    let mut upper_flat = vec![0.0f32; out_size];
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }

    let ctx = BlockedCtx {
        lower: lower_data,
        upper: upper_data,
        weights: &weights,
        row_taps: &row_taps,
        col_taps: &col_taps,
        bias,
        in_c_per_group,
        out_c_per_group,
        kh,
        kw,
        in_w,
        in_spatial,
        out_w,
        gamma,
        abs_inflate,
    };

    // One work unit = one (group, output position); it costs
    // `out_c_per_group · macs` MACs and must stay whole (splitting it would
    // split a reduction). The block admits as many whole units as the budget
    // allows, and never fewer than one.
    let total_units = groups.saturating_mul(spatial);
    let macs_per_unit = out_c_per_group.saturating_mul(macs).max(1);
    let mut units_per_block = (block_macs / macs_per_unit).max(1);
    if parallel {
        units_per_block =
            units_per_block.min((CONV_IBP_PARALLEL_BLOCK_ELEMS / out_c_per_group).max(1));
    }
    units_per_block = units_per_block.min(total_units.max(1));

    let use_parallel = parallel && total_units > 1;
    let mut block_lo = if use_parallel {
        vec![0.0f32; units_per_block * out_c_per_group]
    } else {
        vec![0.0f32; out_c_per_group]
    };
    let mut block_hi = vec![0.0f32; block_lo.len()];
    let mut acc_lo = vec![0.0f64; out_c_per_group];
    let mut acc_hi = vec![0.0f64; out_c_per_group];
    let mut acc_abs = vec![0.0f64; out_c_per_group];

    let mut unit = 0usize;
    while unit < total_units {
        // The ONLY deadline observation on this route, and it sits between
        // whole output elements — so an expiry never publishes a partial fold.
        if Instant::now() >= deadline {
            return Err(deadline_error());
        }
        let end = unit.saturating_add(units_per_block).min(total_units);
        let count = end - unit;
        if use_parallel {
            let width = count * out_c_per_group;
            let (lo_block, hi_block) = (&mut block_lo[..width], &mut block_hi[..width]);
            lo_block
                .par_chunks_mut(out_c_per_group)
                .zip(hi_block.par_chunks_mut(out_c_per_group))
                .enumerate()
                .for_each_init(
                    || {
                        (
                            vec![0.0f64; out_c_per_group],
                            vec![0.0f64; out_c_per_group],
                            vec![0.0f64; out_c_per_group],
                        )
                    },
                    |(scratch_lo, scratch_hi, scratch_abs), (offset, (out_lo, out_hi))| {
                        let index = unit + offset;
                        certified_unit(
                            &ctx,
                            index / spatial,
                            index % spatial,
                            scratch_lo,
                            scratch_hi,
                            scratch_abs,
                            out_lo,
                            out_hi,
                        );
                    },
                );
            for offset in 0..count {
                let index = unit + offset;
                let group = index / spatial;
                let position = index % spatial;
                for oc_local in 0..out_c_per_group {
                    let target = (group * out_c_per_group + oc_local) * spatial + position;
                    lower_flat[target] = block_lo[offset * out_c_per_group + oc_local];
                    upper_flat[target] = block_hi[offset * out_c_per_group + oc_local];
                }
            }
        } else {
            for index in unit..end {
                let group = index / spatial;
                let position = index % spatial;
                certified_unit(
                    &ctx,
                    group,
                    position,
                    &mut acc_lo,
                    &mut acc_hi,
                    &mut acc_abs,
                    &mut block_lo,
                    &mut block_hi,
                );
                for oc_local in 0..out_c_per_group {
                    let target = (group * out_c_per_group + oc_local) * spatial + position;
                    lower_flat[target] = block_lo[oc_local];
                    upper_flat[target] = block_hi[oc_local];
                }
            }
        }
        unit = end;
    }

    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), lower_flat)
        .map_err(|error| {
            NyError::InternalError(format!(
                "conv2d_ibp_forward certified lower reshape: {error}"
            ))
        })?;
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_c, out_h, out_w]), upper_flat)
        .map_err(|error| {
            NyError::InternalError(format!(
                "conv2d_ibp_forward certified upper reshape: {error}"
            ))
        })?;
    if Instant::now() >= deadline {
        return Err(deadline_error());
    }
    Ok(ConvIbpForward {
        lower,
        upper,
        out_h,
        out_w,
    })
}
