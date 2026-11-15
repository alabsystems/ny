// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Compact exact spatial support, keeping dense input/output coordinates.
//!
//! Only rows whose centers AND incoming radii are bitwise zeros are omitted.
//! Their exact contraction is zero, regardless of stored finite weights. All
//! other rows (including zero centers with nonzero uncertainty, cancellation,
//! and subnormal centers/radii) use the existing three checked contractions.
//! Scatter order and the original full receptive-field gamma are retained.
//! A site with no incident active row performs no arithmetic and has exact
//! zero radius; touched sites retain the original underflow floor.
use super::*;

pub(super) fn exact_zero(value: f64) -> bool {
    value.to_bits() & 0x7fff_ffff_ffff_ffff == 0
}

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub oc: usize,
    pub ic: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub ph: usize,
    pub pw: usize,
    pub oh: usize,
    pub ow: usize,
    pub in_h: usize,
    pub in_w: usize,
}

pub(super) fn try_step<E: GemmEngine + ?Sized>(
    eng: &E,
    batch: usize,
    a: &[f64],
    errors: &[f64],
    weights: &[f32],
    g: Geometry,
) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
    // The bounded API has <=8 lower plus <=8 upper rows. Wide frontiers keep
    // their existing path. Dense frontiers also keep the exact old arithmetic.
    if batch > 2 * DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS {
        return Ok(None);
    }
    ny_core::require_f64_interval_proof_environment()?;
    let spatial = g.oh * g.ow;
    let out_dim = g.oc * spatial;
    let in_spatial = g.in_h * g.in_w;
    let in_dim = g.ic * in_spatial;
    let m = batch * spatial;
    let n = g.ic * g.kh * g.kw;
    let mut active = vec![false; m];
    for b in 0..batch {
        for c in 0..g.oc {
            for p in 0..spatial {
                if p % 4096 == 0 {
                    eng.poll_crown_backward_deadline()?;
                }
                let i = b * out_dim + c * spatial + p;
                if !a[i].is_finite() || !errors[i].is_finite() || errors[i] < 0.0 {
                    return Err(NyError::NumericalInstability(
                        "sparse convolution: invalid incoming coefficient/radius".into(),
                    ));
                }
                active[b * spatial + p] |= !exact_zero(a[i]) || !exact_zero(errors[i]);
            }
        }
    }
    let count = active.iter().filter(|&&v| v).count();
    if count > m / 2 {
        return Ok(None);
    }
    // Never let an empty support hide malformed/non-finite source weights.
    for chunk in weights.chunks(4096) {
        eng.poll_crown_backward_deadline()?;
        if chunk.iter().any(|w| !w.is_finite()) {
            return Err(NyError::NumericalInstability(
                "sparse convolution: non-finite weight".into(),
            ));
        }
    }
    if count == 0 {
        eng.poll_crown_backward_deadline()?;
        return Ok(Some((vec![0.0; batch * in_dim], vec![0.0; batch * in_dim])));
    }
    let mut indices = Vec::with_capacity(count);
    let mut ar = Vec::with_capacity(count * g.oc);
    let mut er = Vec::with_capacity(count * g.oc);
    for (row, &present) in active.iter().enumerate() {
        if row % 4096 == 0 {
            eng.poll_crown_backward_deadline()?;
        }
        if !present {
            continue;
        }
        indices.push(row);
        let (b, p) = (row / spatial, row % spatial);
        for c in 0..g.oc {
            let src = b * out_dim + c * spatial + p;
            ar.push(a[src]);
            er.push(errors[src]);
        }
    }
    drop(active);
    let w: Vec<f64> = weights.iter().map(|&v| f64::from(v)).collect();
    let abs_w: Vec<f64> = w.iter().map(|v| v.abs()).collect();
    let abs_a: Vec<f64> = ar.iter().map(|v| v.abs()).collect();
    eng.poll_crown_backward_deadline()?;
    let [ac, sc, pc] =
        eng.gemm_f64_triplet(count, g.oc, n, [&ar, &abs_a, &er], [&w, &abs_w, &abs_w])?;
    eng.poll_crown_backward_deadline()?;
    let mut out = vec![0.0; batch * in_dim];
    let mut mag = vec![0.0; batch * in_dim];
    let mut prop = vec![0.0; batch * in_dim];
    let mut touched = vec![false; batch * in_spatial];
    for (r, &original) in indices.iter().enumerate() {
        eng.poll_crown_backward_deadline()?;
        let (b, p) = (original / spatial, original % spatial);
        let (y, x) = (p / g.ow, p % g.ow);
        for c in 0..g.ic {
            for ky in 0..g.kh {
                let iy = (y * g.sh + ky) as isize - g.ph as isize;
                if iy < 0 || iy >= g.in_h as isize {
                    continue;
                }
                for kx in 0..g.kw {
                    let ix = (x * g.sw + kx) as isize - g.pw as isize;
                    if ix < 0 || ix >= g.in_w as isize {
                        continue;
                    }
                    let pos = iy as usize * g.in_w + ix as usize;
                    let dst = b * in_dim + c * in_spatial + pos;
                    let src = r * n + c * g.kh * g.kw + ky * g.kw + kx;
                    out[dst] += ac[src];
                    mag[dst] += sc[src];
                    prop[dst] += pc[src];
                    touched[b * in_spatial + pos] = true;
                }
            }
        }
    }
    let gamma = gamma_k_f64(g.oc * g.kh * g.kw);
    let factor = 1.0 + 2.0 * gamma;
    let additive = 8.0 * (g.oc * g.kh * g.kw) as f64 * ETA64;
    let mut err = vec![0.0; batch * in_dim];
    for b in 0..batch {
        for c in 0..g.ic {
            for p in 0..in_spatial {
                if p % 4096 == 0 {
                    eng.poll_crown_backward_deadline()?;
                }
                let i = b * in_dim + c * in_spatial + p;
                if touched[b * in_spatial + p] {
                    err[i] = (gamma * mag[i] + prop[i]) * factor + additive;
                }
            }
        }
    }
    eng.poll_crown_backward_deadline()?;
    Ok(Some((out, err)))
}

#[cfg(test)]
#[path = "sparse_conv_tests.rs"]
mod tests;
