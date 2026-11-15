// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified binary64 backward CROWN over a [`ZonoProgram`], using the
//! zonotope's recorded pre-activation boxes — the Z1 replica's
//! `crown_backward` + `concretize_min`, ported term for term.
//!
//! This is the zonotope core's TEST AND DIAGNOSTIC companion: it reproduces the
//! Z1 root margins (`child_zono_replica_parity_row7` — against Z1 run on ny's
//! binary32 BatchNorm coefficients and their certified radii, which this core
//! charges and Z1's own binary64 BatchNorm does not) and proves the sat
//! control leaves stay open (`child_zono_sat_leaf_never_closes`). The
//! production verdict is still ny's own CROWN rebound / fresh clip, which only
//! CONSUMES the zonotope's published map.
//!
//! Relaxation: the CROWN adaptive lower slope (`1` when `u/(u-l) > 1/2`, else
//! `0`) and the chord upper line, whose intercept is validated at both
//! endpoints and patched upward with an explicit margin. Rounding: every
//! adjoint product charges `gamma_{fan+2} |a| (|A| m + |b|)`, every coefficient
//! scaling `u |a s| m`, every constant `gamma` over its terms; the final
//! concretisation over the one-ulp-widened input box charges its own dot
//! product and is rounded down.

use super::program::{StepOp, ZonoProgram, ZonoTrace};
use super::rounding::{gamma, next_down, next_up, TINY, UNIT};
use super::ZonoRefusal;

/// One linear objective `coeffs . y + constant` over the program OUTPUT.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecQuery {
    /// Coefficient per output element.
    pub coeffs: Vec<f64>,
    /// Constant term.
    pub constant: f64,
}

struct ReluRelax {
    lo_s: Vec<f64>,
    up_s: Vec<f64>,
    up_t: Vec<f64>,
    unstable: Vec<bool>,
    n_unstable: usize,
}

/// Adaptive lower slope and certified upper chord on `[lo, hi]`.
fn relu_relax(lo: &[f64], hi: &[f64]) -> ReluRelax {
    let n = lo.len();
    let mut relax = ReluRelax {
        lo_s: vec![0.0; n],
        up_s: vec![0.0; n],
        up_t: vec![0.0; n],
        unstable: vec![false; n],
        n_unstable: 0,
    };
    for j in 0..n {
        let (l, h) = (lo[j], hi[j]);
        if l >= 0.0 {
            relax.lo_s[j] = 1.0;
            relax.up_s[j] = 1.0;
        } else if h > 0.0 {
            let s = h / (h - l);
            let mut t = next_up(-l * s);
            // The line s z + t must dominate relu at both endpoints: s l + t
            // >= 0 and s h + t >= h. Evaluate both, lift the intercept by any
            // shortfall plus a margin covering the evaluation's own rounding.
            let v1 = s * l + t;
            let v2 = s * h + t - h;
            let margin = gamma(4) * ((s * l).abs() + t.abs() + h.abs()) * 2.0 + TINY;
            t = next_up(t + (-v1).max(0.0) + (-v2).max(0.0) + margin);
            relax.up_s[j] = s;
            relax.up_t[j] = t;
            relax.lo_s[j] = if s > 0.5 { 1.0 } else { 0.0 };
            relax.unstable[j] = true;
            relax.n_unstable += 1;
        }
    }
    relax
}

/// Certified lower bound of every query over the program's input box.
///
/// `trace` must come from the same program's forward pass with
/// `record_trace: true`; its boxes are certified enclosures of each ReLU
/// input over that box, which is what makes the relaxation valid.
pub fn spec_lower_bounds(
    program: &ZonoProgram,
    trace: &ZonoTrace,
    queries: &[SpecQuery],
) -> Result<Vec<f64>, ZonoRefusal> {
    let steps = program.steps();
    if trace.magnitudes.len() != steps.len() + 1 {
        return Err(ZonoRefusal::Shape(format!(
            "trace holds {} magnitude vectors for {} steps",
            trace.magnitudes.len(),
            steps.len()
        )));
    }
    let q = queries.len();
    let n_out = program.output_len();
    let mut a = Vec::with_capacity(q * n_out);
    let mut beta = Vec::with_capacity(q);
    for query in queries {
        if query.coeffs.len() != n_out {
            return Err(ZonoRefusal::Shape(format!(
                "query of {} coefficients for {n_out} outputs",
                query.coeffs.len()
            )));
        }
        a.extend_from_slice(&query.coeffs);
        beta.push(query.constant);
    }
    let mut err = vec![0.0f64; q];
    let mut relu_index = trace.pre_boxes.len();
    for (index, step) in steps.iter().enumerate().rev() {
        let m_in = &trace.magnitudes[index];
        match &step.op {
            StepOp::Reshape => {}
            StepOp::Affine(op) => {
                let kernel = &op.kernel;
                let (k_in, k_out) = (kernel.n_in(), kernel.n_out());
                let mut km = vec![0.0f64; k_out];
                kernel.forward_rows(m_in, 1, true, &mut km, None)?;
                let g_adj = gamma(kernel.fan_out() + 2);
                let g_bias = gamma(k_out + 1);
                let weights: Vec<f64> = km
                    .iter()
                    .zip(&op.bias)
                    .map(|(&v, &b)| g_adj * v + g_bias * b.abs())
                    .collect();
                let factor = 1.0 + gamma(k_out + 2);
                for r in 0..q {
                    let row = &a[r * k_out..(r + 1) * k_out];
                    let charge: f64 = row
                        .iter()
                        .zip(&weights)
                        .map(|(x, w)| x.abs() * w)
                        .sum::<f64>()
                        * factor;
                    let bias_term: f64 = row.iter().zip(&op.bias).map(|(x, b)| x * b).sum();
                    let next_beta = beta[r] + bias_term;
                    err[r] += charge + 2.0 * UNIT * (next_beta.abs() + bias_term.abs());
                    beta[r] = next_beta;
                }
                let mut next_a = vec![0.0f64; q * k_in];
                kernel.adjoint_rows(&a, q, false, &mut next_a, None)?;
                a = next_a;
            }
            StepOp::ChannelAffine(op) => {
                let n = op.scale.len();
                let factor = 1.0 + gamma(n + 3);
                let g_shift = gamma(n + 1);
                let weights: Vec<f64> = (0..n)
                    .map(|j| {
                        (UNIT * op.scale[j].abs() + op.scale_err[j]) * m_in[j]
                            + op.shift_err[j]
                            + g_shift * op.shift[j].abs()
                    })
                    .collect();
                for r in 0..q {
                    let row = &mut a[r * n..(r + 1) * n];
                    let charge: f64 = row
                        .iter()
                        .zip(&weights)
                        .map(|(x, w)| x.abs() * w)
                        .sum::<f64>()
                        * factor;
                    let shift_term: f64 = row.iter().zip(&op.shift).map(|(x, t)| x * t).sum();
                    let next_beta = beta[r] + shift_term;
                    err[r] += charge + 2.0 * UNIT * (next_beta.abs() + shift_term.abs());
                    beta[r] = next_beta;
                    for (x, s) in row.iter_mut().zip(&op.scale) {
                        *x *= s;
                    }
                }
            }
            StepOp::Relu => {
                relu_index = relu_index.checked_sub(1).ok_or_else(|| {
                    ZonoRefusal::Shape("trace has fewer ReLU boxes than the program".to_string())
                })?;
                let (lo, hi) = &trace.pre_boxes[relu_index];
                let relax = relu_relax(lo, hi);
                let n = lo.len();
                let coef_factor = UNIT * (1.0 + gamma(relax.n_unstable + 2));
                let const_gamma = gamma(relax.n_unstable + 1);
                for r in 0..q {
                    let row = &mut a[r * n..(r + 1) * n];
                    let mut konst = 0.0f64;
                    let mut abs_konst = 0.0f64;
                    let mut coef_charge = 0.0f64;
                    for j in 0..n {
                        let v = row[j];
                        if v >= 0.0 {
                            row[j] = v * relax.lo_s[j];
                        } else {
                            konst += v * relax.up_t[j];
                            abs_konst += -v * relax.up_t[j];
                            if relax.unstable[j] {
                                coef_charge += -v * relax.up_s[j] * m_in[j];
                            }
                            row[j] = v * relax.up_s[j];
                        }
                    }
                    let next_beta = beta[r] + konst;
                    err[r] += coef_charge * coef_factor
                        + const_gamma * abs_konst
                        + 2.0 * UNIT * (next_beta.abs() + konst.abs());
                    beta[r] = next_beta;
                }
            }
        }
    }
    if relu_index != 0 {
        return Err(ZonoRefusal::Shape(
            "trace has more ReLU boxes than the program".to_string(),
        ));
    }
    let n0 = program.input_len();
    let (xl, xh) = (&trace.input_lo, &trace.input_hi);
    if xl.len() != n0 || xh.len() != n0 {
        return Err(ZonoRefusal::Shape("trace input box length".to_string()));
    }
    let g_final = gamma(n0 + 4);
    let bounds = (0..q)
        .map(|r| {
            let row = &a[r * n0..(r + 1) * n0];
            let mut value = 0.0f64;
            let mut magnitude = 0.0f64;
            for j in 0..n0 {
                let c = row[j];
                value += if c >= 0.0 { c * xl[j] } else { c * xh[j] };
                magnitude += c.abs() * xl[j].abs().max(xh[j].abs());
            }
            let e = err[r] * (1.0 + 1e-9) + TINY;
            let lb = beta[r] + value
                - e
                - g_final * (magnitude + beta[r].abs() + value.abs() + e)
                - TINY;
            next_down(lb)
        })
        .collect();
    Ok(bounds)
}

#[cfg(test)]
mod tests {
    use super::relu_relax;
    use num_rational::BigRational;

    /// The certified upper chord dominates relu at BOTH endpoints in exact
    /// rational arithmetic (a line above a convex function at the endpoints
    /// of an interval is above it on the whole interval), and the adaptive
    /// lower slope is always 0 or 1 — a valid lower line of relu everywhere.
    #[test]
    fn upper_chord_dominates_relu_in_exact_rational_arithmetic() {
        let rat = |v: f64| BigRational::from_float(v).expect("finite");
        let zero = rat(0.0);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            unit
        };
        let mut checked = 0usize;
        for trial in 0..5_000 {
            let scale = 10f64.powi(trial % 11 - 5);
            let lo = -(next() + 1e-3) * scale;
            let hi = (next() + 1e-3) * scale;
            let relax = relu_relax(&[lo], &[hi]);
            assert!(relax.unstable[0]);
            let (s, t) = (rat(relax.up_s[0]), rat(relax.up_t[0]));
            let (l, h) = (rat(lo), rat(hi));
            assert!(
                s.clone() * l + t.clone() >= zero,
                "trial {trial}: chord below relu at l"
            );
            assert!(
                s * h.clone() + t >= h,
                "trial {trial}: chord below relu at h"
            );
            assert!(relax.lo_s[0] == 0.0 || relax.lo_s[0] == 1.0);
            checked += 1;
        }
        assert_eq!(checked, 5_000);
    }
}
