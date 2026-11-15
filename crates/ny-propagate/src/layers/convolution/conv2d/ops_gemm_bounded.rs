// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Optional finite-deadline ConvTranspose CROWN coefficient recomputation.
//!
//! Only the explicit `gemm_f64_with_deadline` contract is used. In particular,
//! this is not permission to hand ordinary CUDA GEMM to a graph verifier.
//! Every output remains a full-k RN-f64 dot product, so the caller's existing
//! cast + gamma_k*S + propagated-error enclosure remains the proof authority.

use super::{Array2, ArrayD, DeadlineInstant, DirectF64PairGeometry, GemmEngine, NyError, Result};
use std::ffi::OsStr;
use std::sync::atomic::{AtomicU64, Ordering};

const POLL_ELEMENTS: usize = 4_096;
const MAX_DISPATCH_MACS: usize = 1 << 24;
// A caller-owned payload ceiling, not a reservation or a whole-process peak.
// Backend/device staging remains under the selected engine's own admission.
// The existing process/Dense guard is also consulted before allocation.
const MAX_OWNED_BYTES: usize = 256 * 1024 * 1024;

type Pair = (Array2<f64>, Array2<f64>);

// Finite authority has its own default-on policy. The old deadline-free gate
// remains unchanged. Literal 0 is the kill switch; malformed values decline.
fn enabled(raw: Option<&OsStr>) -> bool {
    raw.is_none() || raw == Some(OsStr::new("1"))
}

fn report(status: &'static str, backend: &'static str, calls: usize, peak_bytes: usize) {
    static COMPLETIONS: AtomicU64 = AtomicU64::new(0);
    static REFUSALS: AtomicU64 = AtomicU64::new(0);
    let counter = if status == "computed-unpublished" {
        &COMPLETIONS
    } else {
        &REFUSALS
    };
    let event = counter.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    if event <= 4 || event.is_power_of_two() {
        eprintln!(
            "[convtranspose-bounded-f64] status={status} backend={backend} \
            calls_in_computed_pair={calls} caller_owned_bytes={peak_bytes} event={event}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct Quote {
    block_objs: usize,
    rhs_len: usize,
    peak_bytes: usize,
}

fn quote(g: DirectF64PairGeometry) -> Option<Quote> {
    if g.num_objectives == 0 || g.col_width > MAX_DISPATCH_MACS {
        return None;
    }
    let block_objs = g.block_objs.min(g.num_objectives);
    let rows = block_objs.checked_mul(g.spatial_per_obj)?;
    let rhs_len = g.col_width.checked_mul(g.conv_out_c)?;
    // Both complete unpublished results, one im2col block, one returned GEMM
    // block, and one shared RHS. Lower/upper block operands never overlap.
    let elements = g
        .result_len
        .checked_mul(2)?
        .checked_add(rows.checked_mul(g.col_width)?)?
        .checked_add(rows.checked_mul(g.conv_out_c)?)?
        .checked_add(rhs_len)?;
    let peak_bytes = elements.checked_mul(size_of::<f64>())?.checked_add(1_024)?;
    (peak_bytes <= MAX_OWNED_BYTES).then_some(Quote {
        block_objs,
        rhs_len,
        peak_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_pair(
    lower: &Array2<f32>,
    upper: &Array2<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    input_size: (usize, usize),
    deadline: DeadlineInstant,
) -> Result<Option<Pair>> {
    let mut poll = || {
        if DeadlineInstant::now() >= deadline {
            Err(super::super::ops_transpose_gemm::per_node_deadline_exceeded())
        } else {
            Ok(())
        }
    };
    poll()?;
    if crate::sound_f64_gemm::cpu_only_f64_active() {
        return Ok(None);
    }
    if !enabled(
        ny_levers::read_raw_os(&ny_levers::decls::kernel_inputs::CONVTRANSPOSE_SOUND_F64_GPU)
            .as_deref(),
    ) {
        report("policy-off", "not-requested", 0, 0);
        poll()?;
        return Ok(None);
    }
    let Some(g) =
        DirectF64PairGeometry::checked(lower, upper, kernel, stride, padding, dilation, input_size)
    else {
        return Ok(None);
    };
    let Some(q) = quote(g) else {
        report("shape-or-payload-refusal", "not-requested", 0, 0);
        poll()?;
        return Ok(None);
    };
    let macs = q
        .block_objs
        .saturating_mul(g.spatial_per_obj)
        .saturating_mul(g.col_width)
        .saturating_mul(g.conv_out_c);
    if macs < super::super::ops_transpose_gemm::CONV_SOUND_F64_GEMM_MIN_MACS {
        return Ok(None);
    }
    // Check the aggregate caller-owned payload as one f32-sized byte envelope.
    // This is an admission check, not a reservation of backend/device memory.
    // It consults the existing actual Dense/process/host limits; refusal declines
    // this optional route before either engine initialization or allocation.
    if super::super::ops_transpose_gemm::guard_conv_crown_backward_buffer(
        1,
        q.peak_bytes.div_ceil(size_of::<f32>()),
        1,
    )
    .is_err()
    {
        report("envelope-refusal", "not-requested", 0, q.peak_bytes);
        poll()?;
        return Ok(None);
    }
    poll()?;
    let attempt = crate::sound_f64_gemm::with_engine_deadline(deadline, |engine| {
        let result = run(lower, upper, kernel, g, q, engine, deadline, &mut poll);
        if matches!(&result, Ok(None)) {
            report(
                "backend-or-source-refusal",
                engine.backend_provenance(),
                0,
                q.peak_bytes,
            );
        }
        result
    })?;
    if attempt.is_none() {
        report("backend-unavailable", "unavailable", 0, q.peak_bytes);
    }
    poll()?;
    attempt.unwrap_or(Ok(None))
}

fn source_value(value: f32) -> Option<f64> {
    // Refuse source subnormals rather than add a new host-FTZ assumption.
    // Normal f32 products are exactly representable normal f64 values; the
    // inherited gamma_k enclosure then only needs the bounded RN accumulation.
    let bits = value.to_bits() & 0x7fff_ffff;
    if !value.is_finite() || (bits != 0 && bits < 0x0080_0000) {
        None
    } else {
        Some(f64::from(value))
    }
}

fn build(
    len: usize,
    poll: &mut impl FnMut() -> Result<()>,
    mut value: impl FnMut(usize) -> Option<f64>,
) -> Result<Option<Vec<f64>>> {
    poll()?;
    let mut out = Vec::new();
    if out.try_reserve_exact(len).is_err() {
        poll()?;
        return Ok(None);
    }
    poll()?;
    for i in 0..len {
        if i.is_multiple_of(POLL_ELEMENTS) {
            poll()?;
        }
        let Some(x) = value(i) else {
            poll()?;
            return Ok(None);
        };
        out.push(x);
    }
    poll()?;
    Ok(Some(out))
}

#[allow(clippy::too_many_arguments)]
fn product(
    engine: &dyn GemmEngine,
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    deadline: DeadlineInstant,
    poll: &mut impl FnMut() -> Result<()>,
) -> Result<Option<Vec<f64>>> {
    poll()?;
    // No dispatch hint can widen this accelerator cap. The contracted k is
    // unchanged, including padding zeros. Generic f32/f64/pair methods are
    // intentionally unreachable from this finite-authority route.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.gemm_f64_with_deadline(m, k, n, a, b, deadline, MAX_DISPATCH_MACS)
    }));
    let result = match result {
        Ok(result) => {
            poll()?;
            result
        }
        Err(payload) => {
            super::drop_caught_panic_payload(payload);
            poll()?;
            return Ok(None);
        }
    };
    let out = match result {
        Ok(out) if m.checked_mul(n) == Some(out.len()) => out,
        Err(error) if error.is_deadline_exceeded() => return Err(error),
        Ok(_) | Err(_) => return Ok(None),
    };
    for chunk in out.chunks(POLL_ELEMENTS) {
        poll()?;
        if chunk.iter().any(|x| !x.is_finite()) {
            return Ok(None);
        }
    }
    poll()?;
    Ok(Some(out))
}

#[allow(clippy::too_many_arguments)]
fn run(
    lower: &Array2<f32>,
    upper: &Array2<f32>,
    kernel: &ArrayD<f32>,
    g: DirectF64PairGeometry,
    q: Quote,
    engine: &dyn GemmEngine,
    deadline: DeadlineInstant,
    poll: &mut impl FnMut() -> Result<()>,
) -> Result<Option<Pair>> {
    poll()?;
    let Some(rhs) = build(q.rhs_len, poll, |i| {
        let col = i / g.conv_out_c;
        let oc = i % g.conv_out_c;
        let ic = col / g.kernel_spatial;
        let rem = col % g.kernel_spatial;
        source_value(kernel[[oc, ic, rem / g.kw, rem % g.kw]])
    })?
    else {
        return Ok(None);
    };
    let Some(mut lower_result) = build(g.result_len, poll, |_| Some(0.0))? else {
        return Ok(None);
    };
    let Some(mut upper_result) = build(g.result_len, poll, |_| Some(0.0))? else {
        return Ok(None);
    };
    let mut blocks = 0usize;
    let mut obj_start = 0usize;
    while obj_start < g.num_objectives {
        poll()?;
        let block = q.block_objs.min(g.num_objectives - obj_start);
        let rows = block * g.spatial_per_obj;
        for (coefficients, result) in [(lower, &mut lower_result), (upper, &mut upper_result)] {
            let Some(lhs) = build(rows * g.col_width, poll, |i| {
                let pos = i / g.col_width;
                let col = i % g.col_width;
                let obj = obj_start + pos / g.spatial_per_obj;
                let spatial = pos % g.spatial_per_obj;
                let oh = spatial / g.out_w;
                let ow = spatial % g.out_w;
                let ic = col / g.kernel_spatial;
                let rem = col % g.kernel_spatial;
                let ih = oh
                    .checked_mul(g.sh)?
                    .checked_add((rem / g.kw).checked_mul(g.dh)?)?
                    .checked_sub(g.ph)
                    .filter(|&v| v < g.in_h);
                let iw = ow
                    .checked_mul(g.sw)?
                    .checked_add((rem % g.kw).checked_mul(g.dw)?)?
                    .checked_sub(g.pw)
                    .filter(|&v| v < g.in_w);
                match (ih, iw) {
                    (Some(ih), Some(iw)) => {
                        source_value(coefficients[[obj, ic * g.input_spatial + ih * g.in_w + iw]])
                    }
                    _ => Some(0.0),
                }
            })?
            else {
                return Ok(None);
            };
            let Some(output) = product(
                engine,
                rows,
                g.col_width,
                g.conv_out_c,
                &lhs,
                &rhs,
                deadline,
                poll,
            )?
            else {
                return Ok(None);
            };
            for (i, &value) in output.iter().enumerate() {
                if i.is_multiple_of(POLL_ELEMENTS) {
                    poll()?;
                }
                let row = i / g.conv_out_c;
                let oc = i % g.conv_out_c;
                let obj = obj_start + row / g.spatial_per_obj;
                let spatial = row % g.spatial_per_obj;
                result[obj * g.conv_out_size + oc * g.spatial_per_obj + spatial] = value;
            }
            poll()?;
        }
        blocks += 1;
        obj_start += block;
    }
    poll()?;
    let lower = Array2::from_shape_vec((g.num_objectives, g.conv_out_size), lower_result)
        .map_err(|e| NyError::InternalError(format!("bounded ConvTranspose lower shape: {e}")))?;
    let upper = Array2::from_shape_vec((g.num_objectives, g.conv_out_size), upper_result)
        .map_err(|e| NyError::InternalError(format!("bounded ConvTranspose upper shape: {e}")))?;
    poll()?;
    // This proves execution, not publication: this poll and the outer accessor
    // return barrier can still refuse the private pair on late expiry.
    report(
        "computed-unpublished",
        engine.backend_provenance(),
        2 * blocks,
        q.peak_bytes,
    );
    poll()?;
    Ok(Some((lower, upper)))
}

#[cfg(test)]
#[path = "ops_gemm_bounded_tests.rs"]
mod tests;
