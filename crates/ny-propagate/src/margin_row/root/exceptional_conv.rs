// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Absolute error charges for exceptional Conv weight and f64-input casts.
//! Relative gamma bounds miss losses such as 2^-150 casting to zero before
//! multiplication by 2^126. Ordinary conversions preserve existing result bits.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn correct<T: TabElem>(
    c: &ConvOp,
    m: &Array2<T>,
    d: &Array2<T>,
    ginp: &Array2<f64>,
    xabs: &[f64],
    n_in: usize,
    mut gout: Vec<f64>,
) -> Vec<f64> {
    #[cfg(test)]
    if DISABLED.with(std::cell::Cell::get) {
        return gout;
    }
    let has_weight_exception = c.wmat.iter().any(|&w| conversion_gap(w) != 0.0);
    // Native f32 source storage has no f64->f32 input conversion. Its existing
    // ginp already pays a MIN_NORMAL input floor before weight propagation.
    if T::F32_STORE && !has_weight_exception {
        return gout;
    }
    let mut input_extra = vec![0.0; m.nrows()];
    if !T::F32_STORE {
        let widen = next_up(1.0 + next_up(gamma_n_f32(c.k_fwd + 8)));
        input_extra.par_iter_mut().enumerate().for_each(|(j, e)| {
            let mut loss = 0.0;
            for i in 0..=n_in {
                let dm = source_conversion_gap(m[[j, i]].proof_wide());
                let dd = source_conversion_gap(d[[j, i]].proof_wide());
                if dm == 0.0 && dd == 0.0 {
                    continue;
                }
                if !dm.is_finite() || !dd.is_finite() {
                    loss = f64::INFINITY;
                    break;
                }
                let h = if i == n_in { 1.0 } else { next_up(xabs[i]) };
                loss = next_up(loss + next_up(next_up(dm + dd) * h));
            }
            // Also cover arithmetic scaled by the larger rounded input.
            if loss != 0.0 {
                *e = next_up(loss * widen);
            }
        });
    }
    if !has_weight_exception && input_extra.iter().all(|&e| e == 0.0) {
        return gout;
    }
    let delta: Vec<f64> = c.wmat.iter().map(|&w| conversion_gap(w)).collect();
    let mut source = vec![0.0; m.nrows()];
    source.par_iter_mut().enumerate().for_each(|(j, v)| {
        let mut b = 0.0;
        for i in 0..=n_in {
            // Exact bit-level f32 widening retains subnormal stored inputs.
            let a = m[[j, i]].proof_wide().abs();
            let d = d[[j, i]].proof_wide().abs();
            if a == 0.0 && d == 0.0 {
                continue;
            }
            let mag = next_up(a + d);
            let h = if i == n_in { 1.0 } else { next_up(xabs[i]) };
            let term = next_up(mag * h);
            b = next_up(b + term);
        }
        // |q| <= |w| + delta. Besides delta*B, pay delta times the
        // existing incoming/compute/store allowance, so upward subnormal
        // rounding cannot undercharge the arithmetic scaled by |q|.
        *v = next_up(next_up(b + ginp[[j, 0]]) + input_extra[j]);
    });
    let p = c.oshape.1 * c.oshape.2;
    gout.par_iter_mut().enumerate().for_each(|(j, out)| {
        let mut extra = 0.0;
        for t in 0..c.k_fwd {
            let wi = j / p * c.k_fwd + t;
            let gi = c.gather[t * p + j % p];
            if gi == usize::MAX {
                continue;
            }
            let w = c.wmat[wi].abs();
            if w != 0.0 && input_extra[gi] != 0.0 {
                extra = next_up(extra + next_up(w * input_extra[gi]));
            }
            let gap = delta[wi];
            if gap == 0.0 {
                continue;
            }
            if !gap.is_finite() {
                extra = f64::INFINITY;
                break;
            }
            extra = next_up(extra + next_up(gap * source[gi]));
        }
        if extra != 0.0 {
            *out = next_up(*out + extra);
        }
    });
    gout
}

// A source's nonzero f64 subnormal is commonly an ordinary rounding-floor
// D entry. Bound its conversion by a normal value so DAZ cannot erase the
// coefficient gap, without refusing all such rows. For finite accepted
// weights |W| < 2^128, any DAZ-underflowed contribution here is below the
// shared f32 output floor: 2*2^-1022*2^128 per tap and original coordinate,
// versus 2*(k+2)*2^-149 per coordinate. Subnormal f64 weights and captured
// weight overflow instead refuse below, including the huge-source case.
fn source_conversion_gap(value: f64) -> f64 {
    let magnitude_bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
    if magnitude_bits != 0 && magnitude_bits < 0x0010_0000_0000_0000 {
        return f64::MIN_POSITIVE;
    }
    conversion_gap(value)
}

// Relative conversion is valid for a finite normal captured value. Below
// MIN_NORMAL, nearest q has the same sign and |w-q| <= |w|, including a tie
// that rounds upward to twice |w|. Flushing q to zero is covered by |w| too.
// Use bits to detect a captured infinity; never widen a subnormal through DAZ.
fn conversion_gap(w: f64) -> f64 {
    let magnitude_bits = w.to_bits() & 0x7fff_ffff_ffff_ffff;
    if magnitude_bits == 0 {
        return 0.0;
    }
    // A nonzero f64 subnormal can compare equal to zero under DAZ. Do not
    // erase it: even an extremely small original weight can multiply a huge
    // stored f64 source whose f32 conversion overflows. Such operands are
    // outside the finite normal proof arithmetic here; fail closed before
    // any floating-point comparison or product can flush the obligation.
    if magnitude_bits < 0x0010_0000_0000_0000 {
        return f64::INFINITY;
    }
    let q = w as f32;
    if q.is_normal() && w.is_finite() {
        return 0.0;
    }
    if !w.is_finite() || q.to_bits() & 0x7f80_0000 == 0x7f80_0000 {
        return f64::INFINITY;
    }
    next_up(w.abs())
}

#[cfg(test)]
thread_local! {
    static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn without_correction<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(DISABLED.with(|s| s.replace(true)));
    f()
}

#[cfg(test)]
mod tests;
