// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Element-wise and data-movement transformers for DAG zonotope programs
//! (V4 `#fwd-root-screen`, the ml4acopf power-flow graphs).
//!
//! Every transformer here keeps the [`ZonoState`] invariant: if the TRUE
//! value of each input lies in its represented set for the current symbol
//! assignment, the true output lies in the output's represented set for the
//! SAME assignment of every pre-existing symbol and SOME value of each fresh
//! symbol. Rounding is charged with explicit envelopes into the remainder and
//! every endpoint is stepped outward (the core's contract, `mod.rs`).
//!
//! * [`select`] — gather/concat/slice/transpose: output element `j` is an
//!   element of one of several inputs (or a constant). EXACT copies; rows of
//!   different inputs are united BY SYMBOL ID, never by position (plan U8).
//! * [`element_affine`] — `y = a x + t` or `y = x / d + t` per element.
//! * [`product`] — `a * b` of two forms over the same elements: the
//!   first-order terms are the exact ID-aligned products, the second-order
//!   term is bounded by `rad_a * rad_b` and carried by one fresh symbol per
//!   element (the ZONO_NODIAG policy the F1 gate measured).
//! * [`square`] — `x * x`: `(c + s)^2 = c^2 + 2 c s + s^2`, `s^2 in [0, rad^2]`.
//! * [`apply_band`] — any rigorous affine band `lambda x + [mu_lo, mu_hi]`
//!   (Sigmoid/Sin/Cos, Floor, the piecewise-linear lookup tables), with one
//!   fresh symbol per element for the band's half-width.
//! * [`smooth_band`], [`floor_band`], [`pwl_band`] — the band constructors.
//!
//! Test-only fault switches (`fault`) exist solely so the mutation pins can
//! prove the pointwise form checks catch a sign-slipped product and a
//! positional (not ID-aligned) row merge.

use std::collections::HashMap;

use super::ids::{SymbolAllocator, SymbolId};
use super::rounding::{next_down, next_up, TINY, UNIT};
use super::state::ZonoState;
use super::ZonoRefusal;
use crate::network::{pwl_point_enclosure, pwl_range_f64};

/// Where one output element of a [`select`] comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectSource {
    /// Element `elem` of live input `input`.
    Live {
        /// Index into the selection's input list.
        input: usize,
        /// Flat element index within that input.
        elem: usize,
    },
    /// A constant known only as the interval `[lo, hi]` (both exact binary64
    /// images of the model's `f32` constant; `lo == hi` for a point).
    Const {
        /// Lower end.
        lo: f64,
        /// Upper end.
        hi: f64,
    },
}

/// `(product cross-sign slip, positional select)`; both `false` outside tests.
#[cfg(test)]
fn fault_flags() -> (bool, bool) {
    let armed = fault::get();
    (
        armed == fault::ElementwiseFault::ProductCrossSign,
        armed == fault::ElementwiseFault::PositionalSelect,
    )
}

/// `(product cross-sign slip, positional select)`: always `(false, false)`.
#[cfg(not(test))]
#[inline]
const fn fault_flags() -> (bool, bool) {
    (false, false)
}

/// Remove generator rows that are identically zero (an exact operation: a
/// zero row contributes nothing for any symbol value). Selections and dead
/// ReLUs create such rows; dropping them keeps the dense matrix compact.
#[must_use]
pub fn drop_zero_rows(state: ZonoState) -> ZonoState {
    let n = state.len();
    let k = state.num_generators();
    if n == 0 || k == 0 {
        return state;
    }
    let keep: Vec<bool> = (0..k)
        .map(|row| state.generator_row(row).iter().any(|&v| v != 0.0))
        .collect();
    if keep.iter().all(|&kept| kept) {
        return state;
    }
    let (n, center, gens, ids, rem) = state.into_parts();
    let kept = keep.iter().filter(|&&kept| kept).count();
    let mut new_gens = Vec::with_capacity(kept * n);
    let mut new_ids = Vec::with_capacity(kept);
    for (row, &kept_row) in keep.iter().enumerate() {
        if kept_row {
            new_gens.extend_from_slice(&gens[row * n..(row + 1) * n]);
            new_ids.push(ids[row]);
        }
    }
    ZonoState::from_parts_trusted(n, center, new_gens, new_ids, rem)
}

/// Gather output elements from one or more input states (data movement:
/// Slice, Gather, Transpose, Unsqueeze/Squeeze, Reshape/Flatten, Concat).
///
/// EXACT: every centre, generator and remainder entry is copied, never
/// computed. Output rows are the union of the referenced inputs' symbols BY
/// ID (first-seen order); an output element reads only the rows of its own
/// input, so a symbol shared by two inputs keeps its correlation and a symbol
/// only one input knows contributes zero elsewhere. A constant `[lo, hi]`
/// with `lo < hi` becomes centre `mid` with remainder covering the interval.
pub fn select(sources: &[&ZonoState], map: &[SelectSource]) -> Result<ZonoState, ZonoRefusal> {
    let n = map.len();
    let mut used = vec![false; sources.len()];
    for source in map {
        if let SelectSource::Live { input, elem } = *source {
            let Some(state) = sources.get(input) else {
                return Err(ZonoRefusal::Internal(format!(
                    "select reads input {input} of {}",
                    sources.len()
                )));
            };
            if elem >= state.len() {
                return Err(ZonoRefusal::Internal(format!(
                    "select reads element {elem} of an input with {} values",
                    state.len()
                )));
            }
            used[input] = true;
        }
    }
    let (_, positional) = fault_flags();

    let mut row_of: HashMap<SymbolId, usize> = HashMap::new();
    let mut ids: Vec<SymbolId> = Vec::new();
    let mut src_rows: Vec<Vec<usize>> = Vec::with_capacity(sources.len());
    let mut next_extra = sources
        .iter()
        .flat_map(|state| state.ids().iter().map(|id| id.0))
        .max()
        .unwrap_or(0);
    for (input, state) in sources.iter().enumerate() {
        let mut rows = Vec::with_capacity(state.num_generators());
        if used[input] {
            for (row, &id) in state.ids().iter().enumerate() {
                let out_row = if positional {
                    // MUTATION (test only): row `k` of every input shares
                    // output row `k`, labelled with the symbol the FIRST input
                    // holds there — so a later input's row claims the wrong
                    // symbol, exactly the U8 defect shape.
                    if ids.len() <= row {
                        let label = if row_of.contains_key(&id) {
                            next_extra += 1;
                            SymbolId(next_extra)
                        } else {
                            id
                        };
                        row_of.insert(label, ids.len());
                        ids.push(label);
                    }
                    row
                } else {
                    *row_of.entry(id).or_insert_with(|| {
                        ids.push(id);
                        ids.len() - 1
                    })
                };
                rows.push(out_row);
            }
        }
        src_rows.push(rows);
    }

    let k = ids.len();
    let mut gens = vec![
        0.0f64;
        k.checked_mul(n).ok_or_else(|| {
            ZonoRefusal::Shape("select generator size overflows".to_string())
        })?
    ];
    let mut center = vec![0.0f64; n];
    let mut rem = vec![0.0f64; n];
    let mut per_input: Vec<Vec<(usize, usize)>> = vec![Vec::new(); sources.len()];
    for (j, source) in map.iter().enumerate() {
        match *source {
            SelectSource::Live { input, elem } => {
                center[j] = sources[input].center()[elem];
                rem[j] = sources[input].remainder()[elem];
                per_input[input].push((j, elem));
            }
            SelectSource::Const { lo, hi } => {
                if !(lo.is_finite() && hi.is_finite()) || lo > hi {
                    return Err(ZonoRefusal::NonFinite("select constant"));
                }
                if lo == hi {
                    center[j] = lo;
                } else {
                    let mid = 0.5 * lo + 0.5 * hi;
                    let rad = next_up(next_up(hi - mid).max(next_up(mid - lo)));
                    center[j] = mid;
                    rem[j] = next_up(rad + TINY);
                }
            }
        }
    }
    for (input, state) in sources.iter().enumerate() {
        if per_input[input].is_empty() {
            continue;
        }
        for (row, &out_row) in src_rows[input].iter().enumerate() {
            let src = state.generator_row(row);
            let dst = &mut gens[out_row * n..(out_row + 1) * n];
            for &(j, elem) in &per_input[input] {
                dst[j] = src[elem];
            }
        }
    }
    Ok(drop_zero_rows(ZonoState::from_parts_trusted(
        n, center, gens, ids, rem,
    )))
}

/// Gather the elements `map[j]` of ONE input (a single-source [`select`]).
pub fn gather(state: &ZonoState, map: &[usize]) -> Result<ZonoState, ZonoRefusal> {
    let sources: Vec<SelectSource> = map
        .iter()
        .map(|&elem| SelectSource::Live { input: 0, elem })
        .collect();
    select(&[state], &sources)
}

/// The scale of an [`element_affine`] map.
#[derive(Debug, Clone, PartialEq)]
pub enum ElementScale {
    /// `y_j = x_j * a_j + t_j`.
    Mul(Vec<f64>),
    /// `y_j = x_j / d_j + t_j` (`d_j != 0`).
    Div(Vec<f64>),
}

impl ElementScale {
    fn len(&self) -> usize {
        match self {
            Self::Mul(v) | Self::Div(v) => v.len(),
        }
    }
}

/// Rebuild `(n, centre, gens, ids)` with a remainder computed from the NEW
/// generators' certified absolute sums: `rem_j = f(base_j, abs_g_j)`.
fn with_remainder<F>(
    n: usize,
    center: Vec<f64>,
    gens: Vec<f64>,
    ids: Vec<SymbolId>,
    mut rem_of: F,
) -> ZonoState
where
    F: FnMut(usize, f64) -> f64,
{
    let tmp = ZonoState::from_parts_trusted(n, center, gens, ids, vec![0.0; n]);
    let abs_g = tmp.abs_generator_sum();
    let (n, center, gens, ids, _) = tmp.into_parts();
    let rem: Vec<f64> = abs_g
        .iter()
        .enumerate()
        .map(|(j, &g)| rem_of(j, g))
        .collect();
    ZonoState::from_parts_trusted(n, center, gens, ids, rem)
}

/// Per-element affine map with exact binary64 constants (the images of the
/// model's `f32` constants): `y = a x + t` or `y = x / d + t`.
///
/// SOUND. For `x = c + sum G_s e_s + rho`, `a x + t = (a c + t) + sum (a
/// G_s) e_s + a rho`. The computed centre `fl(fl(a c) + t)` is within
/// `4u (|fl(a c)| + |t|)` of `a c + t`; the scaled rows are each within
/// `u |a G_s|`, in total at most `2u * abs_g'` where `abs_g'` is the certified
/// absolute sum of the NEW rows; `|a rho| <= next_up(|a| r)`. Division is the
/// same with `a = 1/d` evaluated as a correctly rounded quotient. The new
/// remainder is the sum of those three terms times `(1 + 4u)`, rounded up.
pub fn element_affine(
    x: &ZonoState,
    scale: &ElementScale,
    shift: &[f64],
) -> Result<ZonoState, ZonoRefusal> {
    let n = x.len();
    if scale.len() != n || shift.len() != n {
        return Err(ZonoRefusal::Shape(format!(
            "element affine of {} scales / {} shifts for {n} values",
            scale.len(),
            shift.len()
        )));
    }
    let mut center = Vec::with_capacity(n);
    let mut base = Vec::with_capacity(n);
    for j in 0..n {
        let (scaled, r_scaled) = match scale {
            ElementScale::Mul(a) => (x.center()[j] * a[j], next_up(a[j].abs() * x.remainder()[j])),
            ElementScale::Div(d) => {
                if d[j] == 0.0 || !d[j].is_finite() {
                    return Err(ZonoRefusal::NonFinite("element affine divisor"));
                }
                (x.center()[j] / d[j], next_up(x.remainder()[j] / d[j].abs()))
            }
        };
        center.push(scaled + shift[j]);
        base.push(r_scaled + 4.0 * UNIT * (scaled.abs() + shift[j].abs()));
    }
    let k = x.num_generators();
    let mut gens = Vec::with_capacity(k * n);
    if n > 0 {
        for row in x.generators().chunks_exact(n) {
            match scale {
                ElementScale::Mul(a) => gens.extend(row.iter().zip(a).map(|(g, s)| g * s)),
                ElementScale::Div(d) => gens.extend(row.iter().zip(d).map(|(g, s)| g / s)),
            }
        }
    }
    Ok(with_remainder(
        n,
        center,
        gens,
        x.ids().to_vec(),
        |j, abs_g| next_up((base[j] + 2.0 * UNIT * abs_g) * (1.0 + 4.0 * UNIT) + TINY),
    ))
}

/// Append one fresh symbol per element whose `radii[j] > 0`, with generator
/// `radii[j]` on element `j` only. Returns the state and the fresh count.
pub(super) fn append_fresh(
    state: ZonoState,
    radii: &[f64],
    alloc: &mut SymbolAllocator,
) -> Result<(ZonoState, usize), ZonoRefusal> {
    let n = state.len();
    if radii.len() != n {
        return Err(ZonoRefusal::Internal(format!(
            "{} fresh radii for {n} values",
            radii.len()
        )));
    }
    if radii.iter().any(|r| !r.is_finite() || *r < 0.0) {
        return Err(ZonoRefusal::NonFinite("fresh symbol radius"));
    }
    let fresh = radii.iter().filter(|&&r| r > 0.0).count();
    if fresh == 0 {
        return Ok((state, 0));
    }
    let (n, center, mut gens, mut ids, rem) = state.into_parts();
    gens.reserve(fresh * n);
    ids.reserve(fresh);
    for (j, &radius) in radii.iter().enumerate() {
        if radius > 0.0 {
            let start = gens.len();
            gens.resize(start + n, 0.0);
            gens[start + j] = radius;
            ids.push(alloc.fresh());
        }
    }
    Ok((
        ZonoState::from_parts_trusted(n, center, gens, ids, rem),
        fresh,
    ))
}

/// `a * b` element-wise for two forms over the same elements (both already
/// gathered to the output layout), aligned BY SYMBOL ID.
///
/// SOUND. With `a = c_a + S_a`, `S_a = sum G_a e + rho_a` (and likewise `b`):
/// `a b = c_a c_b + c_b sum G_a e + c_a sum G_b e + c_b rho_a + c_a rho_b +
/// S_a S_b`. The first-order rows are the ID-aligned combination
/// `G_a[s] c_b + c_a G_b[s]` (a symbol only one side knows has a zero row on
/// the other); `|S_a S_b| <= rad_a rad_b` with `rad = abs_g + r` rounded up,
/// carried by ONE fresh symbol per element (never credited as `e^2`);
/// `|c_b rho_a| + |c_a rho_b|` and every rounding go into the remainder:
/// centre `u |c_a c_b|` (charged `2u |fl|`), each row two products and a sum,
/// in total at most `2u(1+u)(|c_b| abs_a + |c_a| abs_b)` (charged `3u`).
/// Returns the state and the number of fresh symbols.
pub fn product(
    a: &ZonoState,
    b: &ZonoState,
    alloc: &mut SymbolAllocator,
) -> Result<(ZonoState, usize), ZonoRefusal> {
    let n = a.len();
    if b.len() != n {
        return Err(ZonoRefusal::Shape(format!(
            "product of {n} values with {} values",
            b.len()
        )));
    }
    let (cross_sign_fault, _) = fault_flags();
    let sign_b = if cross_sign_fault { -1.0f64 } else { 1.0f64 };
    let abs_a = a.abs_generator_sum();
    let abs_b = b.abs_generator_sum();
    a.ensure_finite(&abs_a)?;
    b.ensure_finite(&abs_b)?;
    let ca = a.center();
    let cb = b.center();
    let b_row_of: HashMap<SymbolId, usize> = b
        .ids()
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, row))
        .collect();
    let mut ids = Vec::with_capacity(a.num_generators() + b.num_generators() + n);
    let mut gens = Vec::with_capacity((a.num_generators() + b.num_generators() + n) * n);
    let mut b_used = vec![false; b.num_generators()];
    for (row, &id) in a.ids().iter().enumerate() {
        let ga = a.generator_row(row);
        ids.push(id);
        match b_row_of.get(&id) {
            Some(&brow) => {
                b_used[brow] = true;
                let gb = b.generator_row(brow);
                for j in 0..n {
                    gens.push(ga[j] * cb[j] + sign_b * (ca[j] * gb[j]));
                }
            }
            None => gens.extend((0..n).map(|j| ga[j] * cb[j])),
        }
    }
    for (row, &id) in b.ids().iter().enumerate() {
        if !b_used[row] {
            ids.push(id);
            let gb = b.generator_row(row);
            gens.extend((0..n).map(|j| sign_b * (ca[j] * gb[j])));
        }
    }
    let mut center = Vec::with_capacity(n);
    let mut rem = Vec::with_capacity(n);
    let mut second = Vec::with_capacity(n);
    for j in 0..n {
        let rad_a = next_up(abs_a[j] + a.remainder()[j]);
        let rad_b = next_up(abs_b[j] + b.remainder()[j]);
        second.push(next_up(rad_a * rad_b));
        let c = ca[j] * cb[j];
        center.push(c);
        let charge =
            2.0 * UNIT * c.abs() + 3.0 * UNIT * (cb[j].abs() * abs_a[j] + ca[j].abs() * abs_b[j]);
        let cross =
            next_up(cb[j].abs() * a.remainder()[j]) + next_up(ca[j].abs() * b.remainder()[j]);
        rem.push(next_up((cross + charge) * (1.0 + 8.0 * UNIT) + TINY));
    }
    let state = ZonoState::from_parts_trusted(n, center, gens, ids, rem);
    let (state, fresh) = append_fresh(state, &second, alloc)?;
    Ok((drop_zero_rows(state), fresh))
}

/// `x * x` element-wise (ONNX `Pow(x, 2)` and `Mul(x, x)`).
///
/// SOUND. `x = c + s`, `s = sum G e + rho`, `|s| <= rad = abs_g + r`:
/// `x^2 = c^2 + 2c sum G e + 2c rho + s^2` with `s^2 in [0, q]`,
/// `q = next_up(rad^2)`. So the centre is `c^2 + h` and one fresh symbol per
/// element carries `h`, where `2h >= q` (`h = q/2`, bumped if the halving was
/// inexact). `|2c rho| <= next_up(2|c| r)`; the centre is within
/// `4u (fl(c^2) + h)` and the rows (`fl(2c)` exact, one product each) within
/// `2u |c| abs_g` (charged `4u`). Returns the state and the fresh count.
pub fn square(
    x: &ZonoState,
    alloc: &mut SymbolAllocator,
) -> Result<(ZonoState, usize), ZonoRefusal> {
    let n = x.len();
    let abs_g = x.abs_generator_sum();
    x.ensure_finite(&abs_g)?;
    let mut center = Vec::with_capacity(n);
    let mut rem = Vec::with_capacity(n);
    let mut half = Vec::with_capacity(n);
    for j in 0..n {
        let c = x.center()[j];
        let rad = next_up(abs_g[j] + x.remainder()[j]);
        let q = next_up(rad * rad);
        let mut h = q * 0.5;
        if h * 2.0 < q {
            h = next_up(h);
        }
        half.push(h);
        let cc = c * c;
        center.push(cc + h);
        let charge = 4.0 * UNIT * (cc + h) + 4.0 * UNIT * c.abs() * abs_g[j];
        let cross = next_up(2.0 * c.abs() * x.remainder()[j]);
        rem.push(next_up((cross + charge) * (1.0 + 8.0 * UNIT) + TINY));
    }
    let k = x.num_generators();
    let mut gens = Vec::with_capacity(k * n);
    if n > 0 {
        for row in x.generators().chunks_exact(n) {
            gens.extend(row.iter().zip(x.center()).map(|(g, c)| g * (2.0 * c)));
        }
    }
    let state = ZonoState::from_parts_trusted(n, center, gens, x.ids().to_vec(), rem);
    let (state, fresh) = append_fresh(state, &half, alloc)?;
    Ok((drop_zero_rows(state), fresh))
}

/// A rigorous per-element affine band: for every `x` in the element's
/// enclosure, `lambda x + mu_lo <= f(x) <= lambda x + mu_hi` in exact real
/// arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    /// Slope per element (any finite value is sound; the offsets are computed
    /// for exactly this binary64 slope).
    pub lambda: Vec<f64>,
    /// Lower offset per element.
    pub mu_lo: Vec<f64>,
    /// Upper offset per element.
    pub mu_hi: Vec<f64>,
}

impl Band {
    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lambda.len()
    }

    /// True for an empty band.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lambda.is_empty()
    }
}

/// Push a state through a [`Band`] that holds on an enclosure of every TRUE
/// input value (the caller's `[lo, hi]`, e.g. the concretisation intersected
/// with ny's IBP box).
///
/// SOUND. For the true `x`, `f(x) = lambda x + mc + delta e_new` for some
/// `e_new in [-1, 1]`, where `mc = (mu_lo + mu_hi)/2` as computed and `delta`
/// rounds `max(mu_hi - mc, mc - mu_lo)` up, so `[mc - delta, mc + delta]`
/// contains `[mu_lo, mu_hi]`. `lambda x + mc` is an [`element_affine`] map
/// (charged there). A zero-width band adds no symbol. Returns the state and
/// the number of fresh symbols.
pub fn apply_band(
    x: &ZonoState,
    band: &Band,
    alloc: &mut SymbolAllocator,
) -> Result<(ZonoState, usize), ZonoRefusal> {
    let n = x.len();
    if band.lambda.len() != n || band.mu_lo.len() != n || band.mu_hi.len() != n {
        return Err(ZonoRefusal::Shape(format!(
            "band of {} elements for {n} values",
            band.lambda.len()
        )));
    }
    let mut mid = Vec::with_capacity(n);
    let mut delta = Vec::with_capacity(n);
    for j in 0..n {
        let (lo, hi, lam) = (band.mu_lo[j], band.mu_hi[j], band.lambda[j]);
        if !(lo.is_finite() && hi.is_finite() && lam.is_finite()) || lo > hi {
            return Err(ZonoRefusal::NonFinite("band offsets"));
        }
        if lo == hi {
            mid.push(lo);
            delta.push(0.0);
        } else {
            let mc = 0.5 * lo + 0.5 * hi;
            mid.push(mc);
            delta.push(next_up(next_up(hi - mc).max(next_up(mc - lo))));
        }
    }
    let base = element_affine(x, &ElementScale::Mul(band.lambda.clone()), &mid)?;
    let (state, fresh) = append_fresh(base, &delta, alloc)?;
    Ok((drop_zero_rows(state), fresh))
}

/// The smooth element-wise functions [`smooth_band`] covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothKind {
    /// Logistic sigmoid `1 / (1 + e^-x)`.
    Sigmoid,
    /// `sin x`.
    Sin,
    /// `cos x`.
    Cos,
}

impl SmoothKind {
    /// Stable lowercase token.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::Sigmoid => "sigmoid",
            Self::Sin => "sin",
            Self::Cos => "cos",
        }
    }

    /// The function in binary64 (libm; its error is padded by the caller).
    #[must_use]
    pub fn eval(self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid(x),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
        }
    }

    /// The derivative in binary64.
    fn derivative(self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid_derivative(x),
            Self::Sin => x.cos(),
            Self::Cos => -x.sin(),
        }
    }

    /// An upper bound on `|f''|` over the reals.
    fn curvature_bound(self) -> f64 {
        match self {
            // max |s (1 - s)(1 - 2s)| = 1 / (6 sqrt 3) = 0.0962250...
            Self::Sigmoid => 0.0963,
            Self::Sin | Self::Cos => 1.0,
        }
    }
}

/// `1 / (1 + e^-x)`: relative error a few ulps (the sum is `>= 1`).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `sigma'(x) = e^-|x| / (1 + e^-|x|)^2` — even, and evaluated WITHOUT the
/// `1 - sigma` cancellation, so its relative error stays a few ulps at every
/// `x` (the pad below is relative).
fn sigmoid_derivative(x: f64) -> f64 {
    let t = (-x.abs()).exp();
    let d = 1.0 + t;
    t / (d * d)
}

/// Pieces the smooth band's Taylor cover splits `[l, u]` into (the F1
/// replica's `K`). A power of two, so `w / (2K)` is exact.
pub const SMOOTH_BAND_PIECES: usize = 1024;

/// Relative pad for one libm evaluation plus the few roundings around it
/// (`4e-15 ~ 18u`; glibc's sin/cos/exp are within 1 ulp).
const LIBM_REL_PAD: f64 = 4e-15;

/// Rigorous chord-parallel affine band of a smooth function over `[l, u]`
/// per element — a port of the F1 replica's `smooth_band`.
///
/// `lambda` is the chord slope (any finite value is sound). The offsets bound
/// `g(x) = f(x) - lambda x` over `[l, u]` by covering it with
/// [`SMOOTH_BAND_PIECES`] pieces `[m - h, m + h]` and the second-order Taylor
/// bound on each: `g(x) <= g(m) + |g'(m)| h + M h^2 / 2` (`|g''| = |f''| <=
/// M`). `h` adds `4 ulp(max(|l|, |u|))` to the half-piece so the computed
/// midpoints' rounding and the rounding of `u - l` still leave `[l, u]`
/// covered; `g(m)` and `g'(m)` are padded by a relative `4e-15` for libm and
/// their own roundings, and the final extrema by a relative `1e-15` of the
/// largest magnitude (which dominates the rounding of the sums) plus one ulp.
pub fn smooth_band(kind: SmoothKind, lo: &[f64], hi: &[f64]) -> Result<Band, ZonoRefusal> {
    if lo.len() != hi.len() {
        return Err(ZonoRefusal::Shape(format!(
            "smooth band bounds {} / {}",
            lo.len(),
            hi.len()
        )));
    }
    let n = lo.len();
    let curvature = kind.curvature_bound();
    #[allow(clippy::cast_precision_loss)]
    let two_k = (2 * SMOOTH_BAND_PIECES) as f64;
    let mut band = Band {
        lambda: Vec::with_capacity(n),
        mu_lo: Vec::with_capacity(n),
        mu_hi: Vec::with_capacity(n),
    };
    for j in 0..n {
        let (l, u) = (lo[j], hi[j]);
        if !(l.is_finite() && u.is_finite()) || l > u {
            return Err(ZonoRefusal::NonFinite("smooth band interval"));
        }
        let w = u - l;
        let wide = w > 1e-12 * (1.0 + l.abs());
        let lam = if wide {
            (kind.eval(u) - kind.eval(l)) / w
        } else {
            kind.derivative(0.5 * l + 0.5 * u)
        };
        if !lam.is_finite() {
            return Err(ZonoRefusal::NonFinite("smooth band slope"));
        }
        let half = w / two_k;
        let scale = l.abs().max(u.abs());
        let ulp = next_up(scale) - scale;
        let h = half * (1.0 + 1e-9) + 4.0 * ulp + TINY;
        let mut up_max = f64::NEG_INFINITY;
        let mut lo_min = f64::INFINITY;
        let mut magnitude = 0.0f64;
        for k in 0..SMOOTH_BAND_PIECES {
            #[allow(clippy::cast_precision_loss)]
            let m = l + (2 * k + 1) as f64 * half;
            let fm = kind.eval(m);
            let dfm = kind.derivative(m);
            let lm = lam * m;
            let g = fm - lm;
            let gp = (dfm - lam).abs() + LIBM_REL_PAD * (dfm.abs() + lam.abs());
            let pad = LIBM_REL_PAD * (fm.abs() + lm.abs() + g.abs()) + 1e-300;
            let spread = gp * h + 0.5 * curvature * (h * h);
            let up = g + spread + pad;
            let down = g - spread - pad;
            // `f64::max`/`min` silently IGNORE a NaN operand, so a NaN piece
            // would drop out of the cover instead of poisoning it. Refuse any
            // non-finite piece outright (an infinite one would make the band
            // infinite and refuse below anyway).
            if !(up.is_finite() && down.is_finite()) {
                return Err(ZonoRefusal::NonFinite("smooth band piece"));
            }
            up_max = up_max.max(up);
            lo_min = lo_min.min(down);
            magnitude = magnitude.max(up.abs()).max(down.abs());
        }
        let mu_hi = next_up(up_max + 1e-15 * magnitude);
        let mu_lo = next_down(lo_min - 1e-15 * magnitude);
        if !(mu_lo.is_finite() && mu_hi.is_finite()) || mu_lo > mu_hi {
            return Err(ZonoRefusal::NonFinite("smooth band offsets"));
        }
        band.lambda.push(lam);
        band.mu_lo.push(mu_lo);
        band.mu_hi.push(mu_hi);
    }
    Ok(band)
}

/// Band of `floor` over `[l, u]` per element: the constant `floor(l)` when
/// `floor(l) == floor(u)` (exact), otherwise `x - 1 < floor(x) <= x`, i.e.
/// `lambda = 1`, `[mu_lo, mu_hi] = [-1, 0]`.
pub fn floor_band(lo: &[f64], hi: &[f64]) -> Result<Band, ZonoRefusal> {
    if lo.len() != hi.len() {
        return Err(ZonoRefusal::Shape("floor band bounds".to_string()));
    }
    let mut band = Band {
        lambda: Vec::with_capacity(lo.len()),
        mu_lo: Vec::with_capacity(lo.len()),
        mu_hi: Vec::with_capacity(lo.len()),
    };
    for (&l, &u) in lo.iter().zip(hi) {
        if !(l.is_finite() && u.is_finite()) || l > u {
            return Err(ZonoRefusal::NonFinite("floor band interval"));
        }
        let (fl, fu) = (l.floor(), u.floor());
        if fl == fu {
            band.lambda.push(0.0);
            band.mu_lo.push(fl);
            band.mu_hi.push(fl);
        } else {
            band.lambda.push(1.0);
            band.mu_lo.push(-1.0);
            band.mu_hi.push(0.0);
        }
    }
    Ok(band)
}

/// Exact affine band of the piecewise-linear lookup table
/// `f(t) = bias + sum_k w_k max(0, t - b_k)` over `[l, u]` per element.
///
/// `g(t) = f(t) - lambda t` is continuous and affine between breakpoints, so
/// its extrema over `[l, u]` sit on `{l, u} ∪ {b_k : l < b_k < u}` (all exact
/// binary64 values). At each candidate, C1's outward-rounded point enclosure
/// brackets `f(t)`, `lambda t` is bracketed by one ulp each way, and the
/// difference is rounded outward. `lambda` is the chord slope of the
/// enclosures' midpoints (any finite value is sound).
pub fn pwl_band(
    lo: &[f64],
    hi: &[f64],
    thresholds: &[f32],
    weights: &[f32],
    bias: f32,
) -> Result<Band, ZonoRefusal> {
    if lo.len() != hi.len() || thresholds.len() != weights.len() || thresholds.is_empty() {
        return Err(ZonoRefusal::Shape("lookup-table band inputs".to_string()));
    }
    if !bias.is_finite()
        || !thresholds
            .iter()
            .chain(weights.iter())
            .all(|v| v.is_finite())
    {
        return Err(ZonoRefusal::NonFinite("lookup-table constants"));
    }
    let mut band = Band {
        lambda: Vec::with_capacity(lo.len()),
        mu_lo: Vec::with_capacity(lo.len()),
        mu_hi: Vec::with_capacity(lo.len()),
    };
    for (&l, &u) in lo.iter().zip(hi) {
        if !(l.is_finite() && u.is_finite()) || l > u {
            return Err(ZonoRefusal::NonFinite("lookup-table band interval"));
        }
        let f_l = pwl_point_enclosure(l, thresholds, weights, bias);
        let f_u = pwl_point_enclosure(u, thresholds, weights, bias);
        let mut lam = if u > l {
            ((0.5 * f_u.0 + 0.5 * f_u.1) - (0.5 * f_l.0 + 0.5 * f_l.1)) / (u - l)
        } else {
            0.0
        };
        if !lam.is_finite() {
            lam = 0.0;
        }
        let mut g_hi = f64::NEG_INFINITY;
        let mut g_lo = f64::INFINITY;
        let mut visit = |t: f64, (f_lo, f_hi): (f64, f64)| {
            let lt = lam * t;
            let (lt_lo, lt_hi) = (next_down(lt), next_up(lt));
            g_hi = g_hi.max(next_up(f_hi - lt_lo));
            g_lo = g_lo.min(next_down(f_lo - lt_hi));
        };
        visit(l, f_l);
        visit(u, f_u);
        for &b in thresholds {
            let bf = f64::from(b);
            if bf > l && bf < u {
                visit(bf, pwl_point_enclosure(bf, thresholds, weights, bias));
            }
        }
        if !(g_lo.is_finite() && g_hi.is_finite()) || g_lo > g_hi {
            return Err(ZonoRefusal::NonFinite("lookup-table band offsets"));
        }
        band.lambda.push(lam);
        band.mu_lo.push(g_lo);
        band.mu_hi.push(g_hi);
    }
    Ok(band)
}

/// C1's exact range of the lookup table over `[l, u]` per element (outward),
/// or an unbounded pair where C1 declines (non-finite interval). The f64
/// interval is first rounded OUTWARD to `f32`, which only widens it.
#[must_use]
pub fn pwl_exact_range(
    lo: &[f64],
    hi: &[f64],
    thresholds: &[f32],
    weights: &[f32],
    bias: f32,
) -> (Vec<f64>, Vec<f64>) {
    lo.iter()
        .zip(hi)
        .map(|(&l, &u)| {
            let (l32, u32) = (ny_core::f64_to_f32_down(l), ny_core::f64_to_f32_up(u));
            pwl_range_f64(l32, u32, thresholds, weights, bias)
                .unwrap_or((f64::NEG_INFINITY, f64::INFINITY))
        })
        .unzip()
}

/// Test-only fault injection for the V4 mutation pins.
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::Cell;

    /// The mutations the pins must catch.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum ElementwiseFault {
        /// Production behaviour.
        None,
        /// [`super::product`]: the classic bilinear sign slip — the second
        /// factor's first-order term enters with the WRONG sign
        /// (`G_a c_b - c_a G_b`).
        ProductCrossSign,
        /// [`super::select`]: merge the rows of different inputs by POSITION
        /// (row `k` of every input lands on output row `k`), the U8 defect.
        PositionalSelect,
    }

    thread_local! {
        static FAULT: Cell<ElementwiseFault> = const { Cell::new(ElementwiseFault::None) };
    }

    /// Arm (or, with `ElementwiseFault::None`, disarm) a mutation on this
    /// thread.
    pub(crate) fn set(fault: ElementwiseFault) {
        FAULT.with(|cell| cell.set(fault));
    }

    pub(crate) fn get() -> ElementwiseFault {
        FAULT.with(Cell::get)
    }
}
