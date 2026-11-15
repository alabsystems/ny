// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Nonnegative operator majorants for rounding charges. For each compiled
//! convolution tap `t`, `a[t] = max_p mag[gather[t,p]]` (padding contributes
//! zero). Then every spatial row satisfies
//!
//! `sum_t |w[o,t]| mag[gather[t,p]] <= sum_t |w[o,t]| a[t]`.
//!
//! The maximum is exact; the one contraction per output channel uses the
//! same outward gamma enclosure as the full spatial contraction. This is an
//! upper bound on ABSOLUTE PRODUCTS, including when the signed output cancels.
//! It is never substituted for a lower bound or a ReLU relaxation. Actual
//! compiled gathers supply the support, including transposed convolutions;
//! geometry metadata, cached root data and class identity supply no authority.

use std::time::Instant;

use crate::zono_core::kernels::check_deadline;
use crate::zono_core::rounding::gamma;
use crate::zono_core::ZonoRefusal;

use super::super::super::net::ConvOp;
use super::{cert_up, MAX_DIM};

#[derive(Debug, Clone, Default)]
pub(crate) struct MagnitudeWork {
    pub(crate) envelope_convs: u64,
    pub(crate) products: u64,
    pub(crate) direct_products: u64,
    pub(crate) gathered: u64,
}

// Polling grain only; neither admission nor a mathematical bound reads it.
const POLL: usize = 4096;

fn shape_error() -> ZonoRefusal {
    ZonoRefusal::Shape("frozen-dual convolution magnitude shape".to_string())
}

pub(super) fn conv_magnitude(
    cv: &ConvOp,
    mag: &[f64],
    deadline: Option<Instant>,
    work: &mut MagnitudeWork,
) -> Result<Vec<f64>, ZonoRefusal> {
    check_deadline(deadline)?;
    let (co, ci, kh, kw) = cv.kernel;
    let k = ci
        .checked_mul(kh)
        .and_then(|v| v.checked_mul(kw))
        .ok_or_else(shape_error)?;
    let p = cv
        .oshape
        .1
        .checked_mul(cv.oshape.2)
        .ok_or_else(shape_error)?;
    let n_out = co.checked_mul(p).ok_or_else(shape_error)?;
    let n_in = cv
        .ishape
        .0
        .checked_mul(cv.ishape.1)
        .and_then(|v| v.checked_mul(cv.ishape.2))
        .ok_or_else(shape_error)?;
    let n_gather = k.checked_mul(p).ok_or_else(shape_error)?;
    let n_weight = co.checked_mul(k).ok_or_else(shape_error)?;
    if cv.k_fwd != k
        || cv.oshape.0 != co
        || cv.ishape.0 != ci
        || cv.gather.len() != n_gather
        || cv.wmat.len() != n_weight
        || mag.len() != n_in
        || k >= MAX_DIM
        || n_out >= MAX_DIM
    {
        return Err(shape_error());
    }
    for chunk in mag.chunks(POLL) {
        check_deadline(deadline)?;
        if chunk.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(ZonoRefusal::NonFinite("frozen-dual input magnitude"));
        }
    }
    for chunk in cv.wmat.chunks(POLL) {
        check_deadline(deadline)?;
        if chunk.iter().any(|v| !v.is_finite()) {
            return Err(ZonoRefusal::NonFinite("frozen-dual magnitude weights"));
        }
    }
    let direct = (n_weight as u64).saturating_mul(p as u64);
    let envelope_work = (n_gather as u64).saturating_add(n_weight as u64);
    let use_envelope = envelope_work < direct;
    let mut result = vec![0.0; n_out];
    let g = gamma(k + 2);
    if use_envelope {
        let mut tap_max = vec![0.0f64; k];
        for (tap, maximum) in tap_max.iter_mut().enumerate() {
            for indices in cv.gather[tap * p..(tap + 1) * p].chunks(POLL) {
                check_deadline(deadline)?;
                for &index in indices {
                    if index != usize::MAX {
                        let value = *mag.get(index).ok_or_else(shape_error)?;
                        *maximum = maximum.max(value);
                    }
                }
            }
        }
        for (o, row) in cv.wmat.chunks(k).enumerate() {
            check_deadline(deadline)?;
            let mut sum = 0.0;
            for (block, weights) in row.chunks(POLL).enumerate() {
                check_deadline(deadline)?;
                for (&weight, &maximum) in weights.iter().zip(&tap_max[block * POLL..]) {
                    sum += weight.abs() * maximum;
                }
            }
            result[o * p..(o + 1) * p].fill(cert_up(sum, g));
        }
    } else {
        // No factorization benefit: retain the original per-row contraction
        // and tap order. This branch also handles empty spatial shapes.
        for o in 0..co {
            for sp in 0..p {
                check_deadline(deadline)?;
                let mut sum = 0.0;
                for tap in 0..k {
                    if tap % POLL == 0 {
                        check_deadline(deadline)?;
                    }
                    let index = cv.gather[tap * p + sp];
                    if index != usize::MAX {
                        let value = *mag.get(index).ok_or_else(shape_error)?;
                        sum += cv.wmat[o * k + tap].abs() * value;
                    }
                }
                result[o * p + sp] = cert_up(sum, g);
            }
        }
    }
    check_deadline(deadline)?;
    if result.iter().any(|v| !v.is_finite()) {
        return Err(ZonoRefusal::NonFinite("frozen-dual convolution magnitude"));
    }
    // Only a completed pass contributes diagnostic counters. No consumer uses
    // them to accept a frame, a state or a prune.
    work.envelope_convs += u64::from(use_envelope);
    work.products = work.products.saturating_add(if use_envelope {
        n_weight as u64
    } else {
        direct
    });
    work.direct_products = work.direct_products.saturating_add(direct);
    work.gathered = work
        .gathered
        .saturating_add(if use_envelope { n_gather as u64 } else { 0 });
    Ok(result)
}

#[cfg(test)]
mod tests;
