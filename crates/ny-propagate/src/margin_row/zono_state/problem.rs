// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Rank-1 split substitution and the CERTIFIED node bound of the frozen dual
//! BaB (port of the P1 replica's `make_problem` / `node_bounds`, which follow
//! Vibecheck's `fast_verify_dual.py:57-110` — minus its far probe, whose
//! production form is unsound by construction, DECISION.md U4b).
//!
//! # Split semantics (exact real arithmetic on the stored floats)
//!
//! A node fixes the sides of the first `D` rows of a split ORDER. For row `i`
//! (neuron `k`, own symbol `enc`, margin coefficient `d_enc = d[enc]`):
//!
//! * `Off` (`z_k <= 0`): `relu = 0`, so `mu e_enc = -lambda z - mu` and
//!   `d_enc e_enc = ro z - d_enc` with `ro = -lambda d_enc / mu`. The concrete
//!   point also satisfies `z <= 0` and (from `e_enc <= 1`, i.e.
//!   `-lambda z <= 2 mu`, when `lambda > 0`) `z >= -2 mu / lambda`.
//! * `On` (`z_k >= 0`): `relu = z`, `d_enc e_enc = rn z - d_enc` with
//!   `rn = (1 - lambda) d_enc / mu`, plus `z >= 0` and (from `e_enc <= 1`, when
//!   `lambda < 1`) `z <= 2 mu / (1 - lambda)`.
//!
//! With `z = c_in + a . e + rho`, `|rho| <= r`, the margin over the node's
//! subdomain is bounded below by the Lagrangian relaxation
//!
//! ```text
//!   g(L) = c0 + sum_i c0s_i - sum_j |rc_j| - sum_i (L0_i b0_i + L1_i b1_i)
//!   rc   = d_base + sum_i (rho_i + sign_i (L0_i - L1_i)) a_i
//! ```
//!
//! (`sign = +1` Off, `-1` On; `d_base` = `d` with every split row's own column
//! zeroed; `b0` the `z`-sign cut and `b1` the far cut, both widened by `r`).
//! `c0s` is the substituted constant ROUNDED DOWN by `|rho| r` (the row
//! remainder), `gamma |rho| |z|max` (the float `rho`'s distance from the exact
//! quotient) and the rounding of `rho c_in - d_enc`. Weak duality makes
//! `g(L)` a valid lower bound for EVERY `L >= 0`, so how a search found `L` is
//! irrelevant; the evaluator clamps `L` to `>= 0` itself and refuses a
//! non-finite one.
//!
//! # The certificate
//!
//! `g` is evaluated in binary64 and published as
//! `next_down(g_f - 1.001 gamma_N S - TINY)`, `N = n + 3D + 32`, with
//! `S = |c0| + sum|c0s| + sum|rc_f| + sum|d| + sum_i (|rho_i| + L0_i + L1_i) |a_i|_1 +
//! sum (L0 |b0| + L1 |b1|)` — the sum of absolute terms of every float
//! operation, order-independent (Higham), so any evaluation order, FMA or a
//! future GEMM kernel is covered. A node may be pruned ONLY on
//! `certified > 0` (strict). The raw value is a search signal and nothing
//! else.

use rayon::prelude::*;

use crate::zono_core::rounding::{gamma, next_down, next_up, TINY, UNIT};
use crate::zono_core::ZonoRefusal;

use super::build::FrozenDualState;
use super::frame::{cert_up, SAFE};

/// Which side of a split a node takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    /// `z <= 0` (the neuron is inactive on the child).
    Off,
    /// `z >= 0` (the neuron is active on the child).
    On,
}

impl Side {
    /// `+1` for `Off`, `-1` for `On` (the Lagrangian's orientation).
    #[must_use]
    pub fn sign(self) -> f64 {
        match self {
            Self::Off => 1.0,
            Self::On => -1.0,
        }
    }
}

/// One node bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeBound {
    /// Raw binary64 `g` at the clamped multipliers (search signal only).
    pub value: f64,
    /// `g` minus the certified rounding charge, rounded down: the ONLY value
    /// that may prune. `-inf` on any refusal.
    pub certified: f64,
    /// The charge subtracted.
    pub charge: f64,
}

impl NodeBound {
    /// The fail-closed bound: never prunes.
    #[must_use]
    pub fn refused() -> Self {
        Self {
            value: f64::NEG_INFINITY,
            certified: f64::NEG_INFINITY,
            charge: f64::INFINITY,
        }
    }

    /// True iff the node is CERTIFIED closed (`certified > 0`, strict).
    #[must_use]
    pub fn prunes(&self) -> bool {
        self.certified > 0.0
    }
}

/// The float data of split row `i` (depth `i` of the order), for callers that
/// re-evaluate a node independently (exact-rational tests, stage B kernels).
#[derive(Debug, Clone, Copy)]
pub struct SplitRow<'a> {
    /// State row index.
    pub state_row: usize,
    /// Own symbol column.
    pub enc: usize,
    /// Row coefficients (`n`).
    pub a: &'a [f64],
    /// Certified upper bound on `sum_j |a_j|`.
    pub row_abs: f64,
    /// Substitution coefficient, Off / On.
    pub rho: [f64; 2],
    /// Certified (rounded-down) substituted constant, Off / On.
    pub c0s: [f64; 2],
    /// `z`-sign cut right-hand side (rounded up), Off / On.
    pub b0: [f64; 2],
    /// Far cut right-hand side (rounded up), Off / On.
    pub b1: [f64; 2],
    /// True when the far cut is only the box bound `sum|a|` (no real cut:
    /// `lambda = 0` Off, `lambda = 1` On). Its multiplier should stay 0.
    pub vacuous: [bool; 2],
}

fn side_index(side: Side) -> usize {
    match side {
        Side::Off => 0,
        Side::On => 1,
    }
}

/// Upper bound on the exact sum of `terms`: the float sum plus
/// `gamma_{n+3} sum|terms|`, stepped up twice.
fn sum_up(terms: &[f64]) -> f64 {
    let s: f64 = terms.iter().sum();
    let a: f64 = terms.iter().map(|v| v.abs()).sum();
    next_up(next_up(s + gamma(terms.len() + 3) * a))
}

/// Lower bound on the exact sum of `terms`.
fn sum_dn(terms: &[f64]) -> f64 {
    let s: f64 = terms.iter().sum();
    let a: f64 = terms.iter().map(|v| v.abs()).sum();
    next_down(next_down(s - gamma(terms.len() + 3) * a))
}

/// The `rc`-independent sums of a node's prologue (see
/// `FrozenDualProblem::prologue_row`), carried from the prologue to the
/// shared epilogue.
#[derive(Debug, Clone, Copy)]
struct NodeSums {
    c0path: f64,
    c0abs: f64,
    lag: f64,
    lagabs: f64,
    tsum: f64,
}

/// Reusable per-thread buffer of the node evaluator.
#[derive(Debug, Clone, Default)]
pub struct NodeScratch {
    rc: Vec<f64>,
}

impl NodeScratch {
    /// The reduced costs `rc` (length `n`) of the LAST
    /// [`FrozenDualProblem::eval_node`] call on this scratch. Complete only
    /// when that call returned a finite `value` (a refusal can leave it
    /// partial or stale). SEARCH SIGNAL ONLY: stage B's line search reads it to
    /// place the breakpoints of the next dual step; no bound is ever read from
    /// it.
    #[must_use]
    pub fn rc(&self) -> &[f64] {
        &self.rc
    }
}

/// Optional subgradient output of [`FrozenDualProblem::eval_node`]: the two
/// cut residuals at the relaxation's minimiser `e* = -sign(rc)`, i.e.
/// `s0_i = sign_i (a_i . e*) - b0_i` and `s1_i = -sign_i (a_i . e*) - b1_i`
/// (a supergradient of `g` in `(L0, L1)`). Search signal only.
pub struct Subgradient<'a> {
    /// `s0` (length `D`).
    pub s0: &'a mut [f64],
    /// `s1` (length `D`).
    pub s1: &'a mut [f64],
}

/// The substituted problem of one state under one split order.
#[derive(Debug, Clone)]
pub struct FrozenDualProblem {
    n: usize,
    order: Vec<usize>,
    a: Vec<f64>,
    row_abs: Vec<f64>,
    enc: Vec<usize>,
    rho: Vec<[f64; 2]>,
    c0s: Vec<[f64; 2]>,
    b0: Vec<[f64; 2]>,
    b1: Vec<[f64; 2]>,
    vacuous: Vec<[bool; 2]>,
    d: Vec<f64>,
    abs_d: f64,
    c0: f64,
    root: f64,
}

impl FrozenDualProblem {
    /// Substitute every row of `order` (distinct state rows; any subset, any
    /// order) into `state`'s margin. The rows are COPIED in split order.
    pub fn new(state: &FrozenDualState, order: &[usize]) -> Result<Self, ZonoRefusal> {
        let n = state.n_gens;
        let s_total = state.num_unstable();
        let mut seen = vec![false; s_total];
        for &k in order {
            if k >= s_total || seen[k] {
                return Err(ZonoRefusal::Internal(format!(
                    "frozen-dual order: row {k} out of range or repeated"
                )));
            }
            seen[k] = true;
        }
        if state.d.len() != n || state.a.len() != s_total * n {
            return Err(ZonoRefusal::Shape("frozen-dual state shape".to_string()));
        }
        let depth = order.len();
        let mut a = Vec::with_capacity(depth * n);
        let mut row_abs = Vec::with_capacity(depth);
        let mut enc = Vec::with_capacity(depth);
        let mut rho = Vec::with_capacity(depth);
        let mut c0s = Vec::with_capacity(depth);
        let mut b0 = Vec::with_capacity(depth);
        let mut b1 = Vec::with_capacity(depth);
        let mut vacuous = Vec::with_capacity(depth);
        for &k in order {
            let row = state.row(k);
            a.extend_from_slice(row);
            let ra = cert_up(row.iter().map(|v| v.abs()).sum::<f64>(), gamma(n + 2));
            row_abs.push(ra);
            let e = state.enc[k];
            enc.push(e);
            let lam = state.lam[k];
            let mu = state.mu[k];
            let cin = state.c_in[k];
            let rk = state.r[k];
            let dnew = state.d[e];
            let zmax = state.lo[k].abs().max(state.hi[k].abs());
            let one_m = 1.0 - lam;
            // rho: the float quotient; its distance from the exact quotient of
            // the stored floats is <= gamma_4 |rho| (Off) / gamma_6 |rho| (On,
            // one more rounding in 1 - lambda), charged against |z| <= zmax.
            let ro = -(lam * dnew) / mu;
            let rn = (one_m * dnew) / mu;
            let c0o_raw = ro * cin - dnew;
            let c0n_raw = rn * cin - dnew;
            let big_ro = (gamma(4) * ro.abs() * zmax
                + ro.abs() * rk * (1.0 + 4.0 * UNIT)
                + gamma(3) * ((ro * cin).abs() + dnew.abs()))
                * SAFE
                + TINY;
            let big_rn = (gamma(6) * rn.abs() * zmax
                + rn.abs() * rk * (1.0 + 4.0 * UNIT)
                + gamma(3) * ((rn * cin).abs() + dnew.abs()))
                * SAFE
                + TINY;
            let c0o = next_down(next_down(c0o_raw - big_ro));
            let c0n = next_down(next_down(c0n_raw - big_rn));
            // Cuts (valid at every concrete point of the child): Off
            //   a.e <= -c_in + r            (z <= 0)
            //  -a.e <=  2mu/lambda + c_in + r  (z >= -2mu/lambda, lambda > 0)
            // On
            //  -a.e <=  c_in + r            (z >= 0)
            //   a.e <=  2mu/(1-lambda) - c_in + r  (z <= 2mu/(1-lambda), lambda < 1)
            // The box bound sum|a| is valid for either far cut at any lambda.
            let b0o = sum_up(&[-cin, rk]);
            let b0n = sum_up(&[cin, rk]);
            let (b1o, vac_o) = if lam > 0.0 {
                let q = next_up(next_up((2.0 * mu) / lam) * (1.0 + 4.0 * UNIT));
                let b = sum_up(&[q, cin, rk]);
                if b.is_finite() {
                    (b.min(ra), false)
                } else {
                    (ra, true)
                }
            } else {
                (ra, true)
            };
            let (b1n, vac_n) = if one_m > 0.0 {
                let q = next_up(next_up((2.0 * mu) / one_m) * (1.0 + 4.0 * UNIT));
                let b = sum_up(&[q, -cin, rk]);
                if b.is_finite() {
                    (b.min(ra), false)
                } else {
                    (ra, true)
                }
            } else {
                (ra, true)
            };
            let vals = [ro, rn, c0o, c0n, b0o, b0n, b1o, b1n, ra];
            if vals.iter().any(|v| !v.is_finite()) {
                return Err(ZonoRefusal::NonFinite("frozen-dual split row"));
            }
            rho.push([ro, rn]);
            c0s.push([c0o, c0n]);
            b0.push([b0o, b0n]);
            b1.push([b1o, b1n]);
            vacuous.push([vac_o, vac_n]);
        }
        let c0 = sum_dn(&[state.c_obj, -state.r_obj]);
        let abs_d_raw: f64 = state.d.iter().map(|v| v.abs()).sum();
        let abs_d = cert_up(abs_d_raw, gamma(n + 2));
        let root = next_down(next_down(
            c0 - abs_d_raw - gamma(n + 4) * (c0.abs() + abs_d) * SAFE - TINY,
        ));
        if !c0.is_finite() || !abs_d.is_finite() || !root.is_finite() {
            return Err(ZonoRefusal::NonFinite("frozen-dual margin"));
        }
        Ok(Self {
            n,
            order: order.to_vec(),
            a,
            row_abs,
            enc,
            rho,
            c0s,
            b0,
            b1,
            vacuous,
            d: state.d.clone(),
            abs_d,
            c0,
            root,
        })
    }

    /// Check an append-only extension of this EXACT numerical problem.
    /// Every value read by any prior node (including all rounding charges)
    /// must match bit for bit. This grants no authority to new rows: those
    /// must still come from the original certified state producer. It only
    /// establishes that old nodes retain their original evaluator.
    pub(crate) fn check_append(
        &self,
        next: &Self,
        deadline: Option<std::time::Instant>,
    ) -> Result<(), ZonoRefusal> {
        let refused = || ZonoRefusal::Internal("frozen-dual append changed prior problem".into());
        let check_time = || {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                Err(ZonoRefusal::Deadline)
            } else {
                Ok(())
            }
        };
        check_time()?;
        let depth = self.max_depth();
        let bits = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        };
        if self.n != next.n
            || next.max_depth() <= depth
            || self.order != next.order[..depth]
            || !bits(&self.d, &next.d)
            || !bits(
                &[self.c0, self.abs_d, self.root],
                &[next.c0, next.abs_d, next.root],
            )
        {
            return Err(refused());
        }
        for i in 0..depth {
            check_time()?;
            if self.enc[i] != next.enc[i]
                || self.vacuous[i] != next.vacuous[i]
                || !bits(
                    &self.a[i * self.n..(i + 1) * self.n],
                    &next.a[i * self.n..(i + 1) * self.n],
                )
                || !bits(&[self.row_abs[i]], &[next.row_abs[i]])
                || !bits(&self.rho[i], &next.rho[i])
                || !bits(&self.c0s[i], &next.c0s[i])
                || !bits(&self.b0[i], &next.b0[i])
                || !bits(&self.b1[i], &next.b1[i])
            {
                return Err(refused());
            }
        }
        check_time()
    }

    /// Generator count `n`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Largest usable depth (`order.len()`).
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.order.len()
    }

    /// State row split at each depth.
    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Certified margin constant `c0 <= c_obj - r_obj`.
    #[must_use]
    pub fn c0(&self) -> f64 {
        self.c0
    }

    /// Margin generator coefficients.
    #[must_use]
    pub fn d(&self) -> &[f64] {
        &self.d
    }

    /// Certified root bound (depth 0): `c0 - sum|d|`, rounded down.
    #[must_use]
    pub fn root_bound(&self) -> f64 {
        self.root
    }

    /// The float data of split row `i`.
    #[must_use]
    pub fn split_row(&self, i: usize) -> SplitRow<'_> {
        SplitRow {
            state_row: self.order[i],
            enc: self.enc[i],
            a: &self.a[i * self.n..(i + 1) * self.n],
            row_abs: self.row_abs[i],
            rho: self.rho[i],
            c0s: self.c0s[i],
            b0: self.b0[i],
            b1: self.b1[i],
            vacuous: self.vacuous[i],
        }
    }

    /// Certified node bound at depth `sides.len()` (see the module docs).
    ///
    /// Multipliers are CLAMPED to `>= 0` here (a NaN clamps to 0); a
    /// non-finite multiplier, a depth beyond [`Self::max_depth`] or a length
    /// mismatch returns [`NodeBound::refused`] (never prunes).
    pub fn eval_node(
        &self,
        sides: &[Side],
        l0: &[f64],
        l1: &[f64],
        scratch: &mut NodeScratch,
        subgrad: Option<Subgradient<'_>>,
    ) -> NodeBound {
        self.eval_node_inner(sides, l0, l1, scratch, subgrad, true)
    }

    /// [`Self::eval_node`] with the certificate optionally DISABLED. The
    /// `charge = false` arm exists only so the mutation pin can prove the
    /// charge is load-bearing; nothing outside tests may call it.
    ///
    /// The per-row PROLOGUE (`prologue_row`) and the scalar EPILOGUE
    /// (`finish_node`) are shared with the block evaluator (`problem/block.rs`),
    /// so the two paths cannot drift apart; only the `rc` accumulation and its
    /// two folds (`sum|rc|` and the subgradient) are laid out differently
    /// there, with every element's operation order preserved.
    pub(crate) fn eval_node_inner(
        &self,
        sides: &[Side],
        l0: &[f64],
        l1: &[f64],
        scratch: &mut NodeScratch,
        subgrad: Option<Subgradient<'_>>,
        charge_on: bool,
    ) -> NodeBound {
        let depth = sides.len();
        if depth > self.order.len() || l0.len() != depth || l1.len() != depth {
            return NodeBound::refused();
        }
        let n = self.n;
        scratch.rc.clear();
        scratch.rc.extend_from_slice(&self.d);
        let rc = &mut scratch.rc;
        for i in 0..depth {
            rc[self.enc[i]] = 0.0;
        }
        let mut sums = self.node_sums_seed();
        for i in 0..depth {
            let Some(mult) = self.prologue_row(i, sides[i], l0[i], l1[i], &mut sums) else {
                return NodeBound::refused();
            };
            if mult != 0.0 {
                let row = &self.a[i * n..(i + 1) * n];
                for (acc, &v) in rc.iter_mut().zip(row) {
                    *acc += mult * v;
                }
            }
        }
        let rc = &scratch.rc;
        let sumabs: f64 = rc.iter().map(|v| v.abs()).sum();
        self.finish_node(
            sides,
            sums,
            sumabs,
            subgrad,
            |i| {
                let row = &self.a[i * n..(i + 1) * n];
                row.iter()
                    .zip(rc.iter())
                    .map(|(&av, &r)| if r < 0.0 { av } else { -av })
                    .sum()
            },
            charge_on,
        )
    }

    /// The prologue sums of a node before its first row.
    #[inline]
    fn node_sums_seed(&self) -> NodeSums {
        NodeSums {
            c0path: self.c0,
            c0abs: self.c0.abs(),
            lag: 0.0f64,
            lagabs: 0.0f64,
            tsum: 0.0f64,
        }
    }

    /// Row `i` of a node's prologue: clamp the multipliers, fold the row's
    /// constant, cut and charge terms into `sums` (in row order, exactly the
    /// expression shapes of the pre-M1 evaluator) and return the row's
    /// multiplier `rho + sign (m0 - m1)` for the `rc` accumulation. `None` on
    /// a non-finite clamped multiplier: the node refuses (`sums` is then
    /// partial and must be discarded).
    #[inline]
    fn prologue_row(
        &self,
        i: usize,
        side: Side,
        l0: f64,
        l1: f64,
        sums: &mut NodeSums,
    ) -> Option<f64> {
        // Clamp: `f64::max` maps NaN to 0.
        let m0 = l0.max(0.0);
        let m1 = l1.max(0.0);
        if !m0.is_finite() || !m1.is_finite() {
            return None;
        }
        let si = side_index(side);
        let sign = side.sign();
        let rho = self.rho[i][si];
        let mult = rho + sign * (m0 - m1);
        let cs = self.c0s[i][si];
        sums.c0path += cs;
        sums.c0abs += cs.abs();
        let b0 = self.b0[i][si];
        let b1 = self.b1[i][si];
        sums.lag += m0 * b0 + m1 * b1;
        sums.lagabs += m0 * b0.abs() + m1 * b1.abs();
        sums.tsum += (rho.abs() + m0 + m1) * self.row_abs[i];
        Some(mult)
    }

    /// The scalar epilogue shared by the node and block evaluators: the raw
    /// value, the optional subgradient (`p_of(i)` is row `i`'s folded
    /// `sum_j (rc_j < 0 ? a_ij : -a_ij)`), the finiteness firewall and the
    /// certificate. `sumabs` is the folded `sum_j |rc_j|`. The subgradient is
    /// written BEFORE the value check, as it always was, so a node whose value
    /// is non-finite still leaves it written.
    #[inline]
    fn finish_node(
        &self,
        sides: &[Side],
        sums: NodeSums,
        sumabs: f64,
        subgrad: Option<Subgradient<'_>>,
        mut p_of: impl FnMut(usize) -> f64,
        charge_on: bool,
    ) -> NodeBound {
        let depth = sides.len();
        let n = self.n;
        let value = sums.c0path - sumabs - sums.lag;
        if let Some(sg) = subgrad {
            if sg.s0.len() == depth && sg.s1.len() == depth {
                for i in 0..depth {
                    let p = p_of(i);
                    let si = side_index(sides[i]);
                    let sign = sides[i].sign();
                    sg.s0[i] = sign * p - self.b0[i][si];
                    sg.s1[i] = -sign * p - self.b1[i][si];
                }
            }
        }
        if !value.is_finite() {
            return NodeBound::refused();
        }
        let big_s = sums.c0abs + sumabs + self.abs_d + sums.tsum + sums.lagabs;
        let charge = if charge_on {
            gamma(n + 3 * depth + 32) * 1.001 * big_s + TINY
        } else {
            0.0
        };
        let certified = next_down(value - charge);
        if !certified.is_finite() || !charge.is_finite() {
            return NodeBound::refused();
        }
        NodeBound {
            value,
            certified,
            charge,
        }
    }

    /// Evaluate `count` nodes of the same `depth` in parallel (bit-identical
    /// to calling [`Self::eval_node`] on each). `sides`, `l0`, `l1` are
    /// row-major `count x depth`; any length mismatch refuses every node.
    #[must_use]
    pub fn eval_batch(
        &self,
        count: usize,
        depth: usize,
        sides: &[Side],
        l0: &[f64],
        l1: &[f64],
    ) -> Vec<NodeBound> {
        let want = count.checked_mul(depth);
        if want != Some(sides.len()) || want != Some(l0.len()) || want != Some(l1.len()) {
            return vec![NodeBound::refused(); count];
        }
        if depth == 0 {
            let mut scratch = NodeScratch::default();
            let one = self.eval_node(&[], &[], &[], &mut scratch, None);
            return vec![one; count];
        }
        sides
            .par_chunks(depth)
            .zip(l0.par_chunks(depth))
            .zip(l1.par_chunks(depth))
            .map_init(NodeScratch::default, |scratch, ((s, a), b)| {
                self.eval_node(s, a, b, scratch, None)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Test-only: the frozen pre-`#fd-block-eval` evaluator and synthetic problems
// ---------------------------------------------------------------------------

#[cfg(test)]
impl FrozenDualProblem {
    /// The node evaluator EXACTLY as it stood before `#fd-block-eval` factored
    /// its prologue and epilogue out (a frozen, verbatim copy). The refactor
    /// pin compares [`Self::eval_node_inner`] against it bit for bit, over
    /// many more inputs than a table of recorded golden values could carry.
    /// Never edit this body: it IS the golden reference.
    pub(crate) fn eval_node_pre_m1(
        &self,
        sides: &[Side],
        l0: &[f64],
        l1: &[f64],
        scratch: &mut NodeScratch,
        subgrad: Option<Subgradient<'_>>,
        charge_on: bool,
    ) -> NodeBound {
        let depth = sides.len();
        if depth > self.order.len() || l0.len() != depth || l1.len() != depth {
            return NodeBound::refused();
        }
        let n = self.n;
        scratch.rc.clear();
        scratch.rc.extend_from_slice(&self.d);
        let rc = &mut scratch.rc;
        for i in 0..depth {
            rc[self.enc[i]] = 0.0;
        }
        let mut c0path = self.c0;
        let mut c0abs = self.c0.abs();
        let mut lag = 0.0f64;
        let mut lagabs = 0.0f64;
        let mut tsum = 0.0f64;
        for i in 0..depth {
            // Clamp: `f64::max` maps NaN to 0.
            let m0 = l0[i].max(0.0);
            let m1 = l1[i].max(0.0);
            if !m0.is_finite() || !m1.is_finite() {
                return NodeBound::refused();
            }
            let si = side_index(sides[i]);
            let sign = sides[i].sign();
            let rho = self.rho[i][si];
            let mult = rho + sign * (m0 - m1);
            if mult != 0.0 {
                let row = &self.a[i * n..(i + 1) * n];
                for (acc, &v) in rc.iter_mut().zip(row) {
                    *acc += mult * v;
                }
            }
            let cs = self.c0s[i][si];
            c0path += cs;
            c0abs += cs.abs();
            let b0 = self.b0[i][si];
            let b1 = self.b1[i][si];
            lag += m0 * b0 + m1 * b1;
            lagabs += m0 * b0.abs() + m1 * b1.abs();
            tsum += (rho.abs() + m0 + m1) * self.row_abs[i];
        }
        let sumabs: f64 = rc.iter().map(|v| v.abs()).sum();
        let value = c0path - sumabs - lag;
        if let Some(sg) = subgrad {
            if sg.s0.len() == depth && sg.s1.len() == depth {
                for i in 0..depth {
                    let row = &self.a[i * n..(i + 1) * n];
                    let p: f64 = row
                        .iter()
                        .zip(rc.iter())
                        .map(|(&av, &r)| if r < 0.0 { av } else { -av })
                        .sum();
                    let si = side_index(sides[i]);
                    let sign = sides[i].sign();
                    sg.s0[i] = sign * p - self.b0[i][si];
                    sg.s1[i] = -sign * p - self.b1[i][si];
                }
            }
        }
        if !value.is_finite() {
            return NodeBound::refused();
        }
        let big_s = c0abs + sumabs + self.abs_d + tsum + lagabs;
        let charge = if charge_on {
            gamma(n + 3 * depth + 32) * 1.001 * big_s + TINY
        } else {
            0.0
        };
        let certified = next_down(value - charge);
        if !certified.is_finite() || !charge.is_finite() {
            return NodeBound::refused();
        }
        NodeBound {
            value,
            certified,
            charge,
        }
    }

    /// A random problem of `n` generators and `min(depth, n)` split rows, for
    /// BIT-IDENTITY and rate tests only (certificate validity is irrelevant
    /// there): finite rows with ~1 in 5 exact zeros, distinct own columns,
    /// `row_abs = cert_up(sum|a|)`, finite `rho` (some exactly `0.0`, so a
    /// zero multiplier's skip is exercised), `c0s`, `b0`, `b1`, random
    /// `vacuous` flags, some `-0.0` margin coefficients, and `abs_d`, `c0`,
    /// `root` computed exactly as [`Self::new`] computes them.
    pub(crate) fn synthetic(n: usize, depth: usize, seed: u64) -> Self {
        Self::synthetic_with(n, depth, seed, false)
    }

    /// [`Self::synthetic`] with every row, cut and constant exactly zero and
    /// every margin coefficient `-0.0`: the folds then see only signed zeros,
    /// which is where a wrong accumulator seed would show.
    pub(crate) fn synthetic_degenerate(n: usize, depth: usize, seed: u64) -> Self {
        Self::synthetic_with(n, depth, seed, true)
    }

    fn synthetic_with(n: usize, depth: usize, seed: u64, degenerate: bool) -> Self {
        let n = n.max(1);
        let depth = depth.min(n);
        let mut rng = SplitMix(seed ^ 0xA076_1D64_78BD_642F);
        // Distinct own columns: a partial Fisher-Yates shuffle of 0..n.
        let mut cols: Vec<usize> = (0..n).collect();
        for i in 0..depth {
            let j = i + rng.below(n - i);
            cols.swap(i, j);
        }
        let enc: Vec<usize> = cols[..depth].to_vec();
        let mut a = vec![0.0f64; depth * n];
        if !degenerate {
            for v in &mut a {
                *v = if rng.below(5) == 0 {
                    0.0
                } else {
                    rng.range(-1.0, 1.0)
                };
            }
        }
        let row_abs: Vec<f64> = (0..depth)
            .map(|i| {
                let s: f64 = a[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum();
                cert_up(s, gamma(n + 2))
            })
            .collect();
        let mut rho = Vec::with_capacity(depth);
        let mut c0s = Vec::with_capacity(depth);
        let mut b0 = Vec::with_capacity(depth);
        let mut b1 = Vec::with_capacity(depth);
        let mut vacuous = Vec::with_capacity(depth);
        for _ in 0..depth {
            if degenerate {
                rho.push([0.0, -0.0]);
                c0s.push([0.0, 0.0]);
                b0.push([0.0, 0.0]);
                b1.push([0.0, 0.0]);
            } else {
                let r = if rng.below(8) == 0 {
                    [0.0, -0.0]
                } else {
                    [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)]
                };
                rho.push(r);
                c0s.push([rng.range(-0.5, 0.0), rng.range(-0.5, 0.0)]);
                b0.push([rng.range(0.0, 3.0), rng.range(0.0, 3.0)]);
                b1.push([rng.range(0.0, 3.0), rng.range(0.0, 3.0)]);
            }
            vacuous.push([rng.below(2) == 0, rng.below(2) == 0]);
        }
        let d: Vec<f64> = (0..n)
            .map(|_| {
                if degenerate || rng.below(7) == 0 {
                    -0.0
                } else {
                    rng.range(-1.0, 1.0)
                }
            })
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let c_obj = if degenerate {
            0.0
        } else {
            rng.range(0.0, 0.5) * n as f64
        };
        let r_obj = if degenerate {
            0.0
        } else {
            rng.range(0.0, 1e-3)
        };
        let c0 = sum_dn(&[c_obj, -r_obj]);
        let abs_d_raw: f64 = d.iter().map(|v| v.abs()).sum();
        let abs_d = cert_up(abs_d_raw, gamma(n + 2));
        let root = next_down(next_down(
            c0 - abs_d_raw - gamma(n + 4) * (c0.abs() + abs_d) * SAFE - TINY,
        ));
        Self {
            n,
            order: (0..depth).collect(),
            a,
            row_abs,
            enc,
            rho,
            c0s,
            b0,
            b1,
            vacuous,
            d,
            abs_d,
            c0,
            root,
        }
    }

    /// A copy with the margin constant replaced (and the root recomputed as
    /// [`Self::new`] would): the rate probe centres synthetic node values on
    /// zero so about half its nodes prune, as on the real rows.
    pub(crate) fn with_c0_for_test(&self, c0: f64) -> Self {
        let mut out = self.clone();
        let abs_d_raw: f64 = out.d.iter().map(|v| v.abs()).sum();
        out.c0 = c0;
        out.root = next_down(next_down(
            c0 - abs_d_raw - gamma(out.n + 4) * (c0.abs() + out.abs_d) * SAFE - TINY,
        ));
        out
    }
}

/// splitmix64: a dependency-free deterministic generator for the synthetic
/// problems (test-only).
#[cfg(test)]
struct SplitMix(u64);

#[cfg(test)]
impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn below(&mut self, k: usize) -> usize {
        (self.next_u64() % (k.max(1) as u64)) as usize
    }

    #[allow(clippy::cast_precision_loss)]
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

// `#fd-block-eval`: the same-depth block form of `FrozenDualProblem::eval_node`,
// bit-identical to it (its own module docs carry the contract).
pub(crate) mod block;

#[cfg(test)]
mod append_tests {
    use super::super::frame::ReluSite;
    use super::*;

    fn state() -> FrozenDualState {
        FrozenDualState {
            t: 0,
            class: 1,
            n_gens: 3,
            n_in_gens: 1,
            c_obj: -2.0,
            d: vec![0.125, 0.25, -0.5],
            r_obj: 0.01,
            a: vec![1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            c_in: vec![0.0, 0.0],
            r: vec![0.01, 0.02],
            lam: vec![0.5, 0.5],
            mu: vec![0.5, 0.5],
            lo: vec![-1.0, -1.0],
            hi: vec![1.0, 1.0],
            enc: vec![1, 2],
            layer: vec![0, 0],
            site: vec![ReluSite::Head; 2],
            neuron: vec![0, 1],
            stats: Default::default(),
        }
    }

    #[test]
    fn checked_append_preserves_every_prior_node_bits_and_polls_deadline() {
        let s = state();
        let old = FrozenDualProblem::new(&s, &[0]).unwrap();
        let new = FrozenDualProblem::new(&s, &[0, 1]).unwrap();
        assert!(old.check_append(&new, None).is_ok());
        for sides in [vec![], vec![Side::Off], vec![Side::On]] {
            for (a, b) in [(0.0, 0.0), (0.25, 0.75), (8.0, 3.0), (-1.0, 2.0)] {
                let l0 = vec![a; sides.len()];
                let l1 = vec![b; sides.len()];
                let x = old.eval_node(&sides, &l0, &l1, &mut NodeScratch::default(), None);
                let y = new.eval_node(&sides, &l0, &l1, &mut NodeScratch::default(), None);
                assert_eq!(x.value.to_bits(), y.value.to_bits());
                assert_eq!(x.certified.to_bits(), y.certified.to_bits());
                assert_eq!(x.charge.to_bits(), y.charge.to_bits());
            }
        }
        assert!(matches!(
            old.check_append(
                &new,
                Some(
                    std::time::Instant::now()
                        .checked_sub(std::time::Duration::from_millis(1))
                        .unwrap()
                )
            ),
            Err(ZonoRefusal::Deadline)
        ));
    }

    #[test]
    #[expect(
        clippy::redundant_clone,
        reason = "Keep the original fixture owner intact while mutating an independent candidate."
    )]
    fn checked_append_rejects_all_prior_numerical_or_layout_mutations() {
        let s = state();
        let old = FrozenDualProblem::new(&s, &[0]).unwrap();
        let pristine = FrozenDualProblem::new(&s, &[0, 1]).unwrap();
        let moved = |v: f64| f64::from_bits(v.to_bits() ^ 1);
        let mut checked = 0;
        let mut reject = |p: FrozenDualProblem| {
            assert!(old.check_append(&p, None).is_err());
            checked += 1;
        };
        let mut p = pristine.clone();
        p.n += 1;
        reject(p);
        let mut p = pristine.clone();
        p.order[0] = 1;
        reject(p);
        let mut p = pristine.clone();
        p.d[0] = moved(p.d[0]);
        reject(p);
        let mut p = pristine.clone();
        p.c0 = moved(p.c0);
        reject(p);
        let mut p = pristine.clone();
        p.abs_d = moved(p.abs_d);
        reject(p);
        let mut p = pristine.clone();
        p.root = moved(p.root);
        reject(p);
        let mut p = pristine.clone();
        p.a[0] = moved(p.a[0]);
        reject(p);
        let mut p = pristine.clone();
        p.enc[0] = 2;
        reject(p);
        let mut p = pristine.clone();
        p.row_abs[0] = moved(p.row_abs[0]);
        reject(p);
        let mut p = pristine.clone();
        p.rho[0][0] = moved(p.rho[0][0]);
        reject(p);
        let mut p = pristine.clone();
        p.c0s[0][0] = moved(p.c0s[0][0]);
        reject(p);
        let mut p = pristine.clone();
        p.b0[0][0] = moved(p.b0[0][0]);
        reject(p);
        let mut p = pristine.clone();
        p.b1[0][0] = moved(p.b1[0][0]);
        reject(p);
        let mut p = pristine.clone();
        p.vacuous[0][0] = !p.vacuous[0][0];
        reject(p);
        reject(old.clone());
        assert_eq!(checked, 15);
    }
}
