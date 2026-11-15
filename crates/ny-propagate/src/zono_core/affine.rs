// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Affine transformers: `y = A x + b` (Gemm / Conv2d / ConvTranspose2d) and
//! the exact per-channel affine of BatchNormalization with its certified
//! coefficient radii.

use std::time::Instant;

use rayon::prelude::*;

use super::kernels::{LinearKernel, RowPar};
use super::rounding::{gamma, next_up, TINY, UNIT};
use super::state::ZonoState;
use super::ZonoRefusal;

/// `y = A x + b` with `b` given per OUTPUT element.
#[derive(Debug, Clone)]
pub struct AffineOp {
    /// The linear part.
    pub kernel: LinearKernel,
    /// Bias per output element (binary64 image of the model's `f32` bias).
    pub bias: Vec<f64>,
}

impl AffineOp {
    /// Build an affine op, checking the bias length.
    pub fn new(kernel: LinearKernel, bias: Vec<f64>) -> Result<Self, ZonoRefusal> {
        if bias.len() != kernel.n_out() {
            return Err(ZonoRefusal::Shape(format!(
                "affine bias of {} values for {} outputs",
                bias.len(),
                kernel.n_out()
            )));
        }
        Ok(Self { kernel, bias })
    }
}

/// Per-element affine `y_j = s_real_j x_j + t_real_j` known only through the
/// stored binary64 `scale`/`shift` and certified radii:
/// `|s_real - scale| <= scale_err`, `|t_real - shift| <= shift_err`.
/// BatchNormalization supplies these from `BatchNormLayer`'s exact-real
/// enclosure (never an uncharged fold, plan U10): the binary64 enclosure
/// itself when the layer carries an intact one, else the binary32 affine and
/// its certified radii (see [`BatchNormCoefficients`]).
#[derive(Debug, Clone)]
pub struct ChannelAffine {
    /// Stored scale per element.
    pub scale: Vec<f64>,
    /// Stored shift per element.
    pub shift: Vec<f64>,
    /// Certified scale radius per element.
    pub scale_err: Vec<f64>,
    /// Certified shift radius per element.
    pub shift_err: Vec<f64>,
}

/// Which representation of a BatchNorm's exact-real affine a compiled
/// [`ChannelAffine`] step charges (#zono-exact-bn). Both satisfy the
/// [`ChannelAffine`] contract; they differ only in how tight it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchNormCoefficients {
    /// Binary64 centres of the outward-rounded binary64 enclosure
    /// `BatchNormLayer::new` computes of the exact-real coefficients, with
    /// radii of a few binary64 ulps (~`1e-15` relative).
    ExactF64,
    /// The layer's binary32 `scale` / `bias` with their certified binary32
    /// radii `scale_err` / `bias_err` (~`6e-8` relative for a `new` layer; the
    /// path every layer took before #zono-exact-bn, and the fallback for a
    /// layer without an intact exact enclosure).
    Binary32,
}

impl BatchNormCoefficients {
    /// Stable lowercase token for observables.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::ExactF64 => "exact_f64",
            Self::Binary32 => "binary32",
        }
    }
}

/// Push a state through `y = A x + b`.
///
/// `abs_g` must be `state.abs_generator_sum()`.
///
/// SOUND. For the true value `x = c + G e + rho`:
/// `A x + b = (A c + b) + (A G) e + A rho`. The computed centre and generator
/// rows differ from the exact ones by at most `gamma_{f+1} (|A| |c| + |b|)`
/// and `gamma_f |A| |G_k|` per row (`f` = fan-in; Higham, any summation
/// order), and `|A rho| <= |A| r`. With `m >= |c| + sum |G| + r`, the new
/// remainder `(|A| r + gamma_{f+2} (|A| m + |b|)) (1 + gamma_{f+5})`, rounded
/// up, dominates all of it; the trailing factor covers the rounding of the two
/// magnitude products themselves.
pub fn apply_affine(
    state: &ZonoState,
    abs_g: &[f64],
    op: &AffineOp,
    deadline: Option<Instant>,
) -> Result<ZonoState, ZonoRefusal> {
    apply_affine_par(state, abs_g, op, deadline, RowPar::Seq).map(|(next, _)| next)
}

/// [`apply_affine`] under a row schedule (#m8-cgan-leaf-batch): ONLY the
/// `K`-row generator product runs under `par` (the reference's own row
/// blocks as rayon tasks, bitwise identical, see [`LinearKernel::forward_rows_par`]);
/// the one-row centre, magnitude and remainder products and every charge are
/// the sequential code unchanged. Returns the state and the generator call's
/// row-block count (telemetry).
pub fn apply_affine_par(
    state: &ZonoState,
    abs_g: &[f64],
    op: &AffineOp,
    deadline: Option<Instant>,
    par: RowPar,
) -> Result<(ZonoState, usize), ZonoRefusal> {
    let kernel = &op.kernel;
    let (n_in, n_out) = (kernel.n_in(), kernel.n_out());
    if state.len() != n_in || abs_g.len() != n_in {
        return Err(ZonoRefusal::Shape(format!(
            "affine op expects {n_in} inputs, state has {}",
            state.len()
        )));
    }
    let k = state.num_generators();
    let m = state.magnitude_par(abs_g, par);

    let mut center = vec![0.0f64; n_out];
    kernel.forward_rows(state.center(), 1, false, &mut center, deadline)?;
    for (c, b) in center.iter_mut().zip(&op.bias) {
        *c += *b;
    }
    let mut am = vec![0.0f64; n_out];
    kernel.forward_rows(&m, 1, true, &mut am, deadline)?;
    let mut ar = vec![0.0f64; n_out];
    kernel.forward_rows(state.remainder(), 1, true, &mut ar, deadline)?;

    let mut gens = vec![0.0f64; k * n_out];
    let row_tasks =
        kernel.forward_rows_par(state.generators(), k, false, &mut gens, deadline, par)?;

    let fan = kernel.fan_in();
    let g_round = gamma(fan + 2);
    let g_eval = 1.0 + gamma(fan + 5);
    let rem: Vec<f64> = ar
        .iter()
        .zip(&am)
        .zip(&op.bias)
        .map(|((&r, &mag), &b)| next_up((r + g_round * (mag + b.abs())) * g_eval + TINY))
        .collect();
    Ok((
        ZonoState::from_parts_trusted(n_out, center, gens, state.ids().to_vec(), rem),
        row_tasks,
    ))
}

/// Push a state through the per-element affine of [`ChannelAffine`].
///
/// SOUND. `s_real x + t_real = (s c + t) + (s G) e + s rho + (s_real - s) x +
/// (t_real - t)`. Centre rounding is at most `2u (|s c| + |t|)`, generator
/// rounding `u |s| |G_k|` per row, the model error `scale_err |x| + shift_err
/// <= scale_err m + shift_err`, and `|s rho| <= |s| r`. The new remainder
/// `(|s| r + (scale_err + 3u|s|) m + shift_err + 3u|t|) (1 + 8u)`, rounded up,
/// dominates all of it (`m >= |c| + sum |G| + r`).
pub fn apply_channel_affine(
    state: &ZonoState,
    abs_g: &[f64],
    op: &ChannelAffine,
) -> Result<ZonoState, ZonoRefusal> {
    apply_channel_affine_par(state, abs_g, op, RowPar::Seq)
}

/// [`apply_channel_affine`] under a row schedule (#m8-cgan-leaf-batch): the
/// `K` generator rows are independent and each entry is the same single
/// product `g * s` as the sequential loop, so [`RowPar::Rayon`] (one task per
/// row) is bitwise identical. Centre, remainder and charges are unchanged.
pub fn apply_channel_affine_par(
    state: &ZonoState,
    abs_g: &[f64],
    op: &ChannelAffine,
    par: RowPar,
) -> Result<ZonoState, ZonoRefusal> {
    let n = state.len();
    if op.scale.len() != n
        || op.shift.len() != n
        || op.scale_err.len() != n
        || op.shift_err.len() != n
        || abs_g.len() != n
    {
        return Err(ZonoRefusal::Shape(format!(
            "channel affine of {} values applied to {n}",
            op.scale.len()
        )));
    }
    let m = state.magnitude_par(abs_g, par);
    let center: Vec<f64> = state
        .center()
        .iter()
        .zip(&op.scale)
        .zip(&op.shift)
        .map(|((&c, &s), &t)| c * s + t)
        .collect();
    let k = state.num_generators();
    let gens = if par == RowPar::Rayon && n > 0 && k > 1 {
        let mut gens = vec![0.0f64; k * n];
        gens.par_chunks_mut(n)
            .zip(state.generators().par_chunks_exact(n))
            .for_each(|(dst, row)| {
                for ((slot, g), s) in dst.iter_mut().zip(row).zip(&op.scale) {
                    *slot = g * s;
                }
            });
        gens
    } else {
        let mut gens = Vec::with_capacity(k * n);
        for row in state.generators().chunks_exact(n.max(1)) {
            gens.extend(row.iter().zip(&op.scale).map(|(g, s)| g * s));
        }
        gens
    };
    let rem: Vec<f64> = (0..n)
        .map(|j| {
            let s = op.scale[j].abs();
            let charge = s * state.remainder()[j]
                + (op.scale_err[j] + 3.0 * UNIT * s) * m[j]
                + op.shift_err[j]
                + 3.0 * UNIT * op.shift[j].abs();
            next_up(charge * (1.0 + 8.0 * UNIT) + TINY)
        })
        .collect();
    Ok(ZonoState::from_parts_trusted(
        n,
        center,
        gens,
        state.ids().to_vec(),
        rem,
    ))
}
