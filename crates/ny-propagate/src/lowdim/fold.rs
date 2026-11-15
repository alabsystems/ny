// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#lowdim-fold` — admit a graph to the lowdim lane by its FREE input
//! coordinates, folding everything that does not depend on them.
//!
//! # Why
//!
//! nn4sys `cardinality_*_2048*` (mscn): every OR-clause is an input box with
//! 1-4 free scalars and 150-300 FIXED coordinates. alpha-beta-CROWN solves
//! these rows (median 17.5 s) by plain input-split BaB with linear bounds over
//! the free coordinates only; ny's box screen walks the whole 2048-wide DAG per
//! node and is ~72x short. The lowdim lane is exactly abc's method but admits
//! only vector-shaped graphs with <= 16 inputs. This module compiles a clause
//! into a lane [`Program`] over the FREE coordinates:
//!
//! * every graph element is either a sound f64 interval CONSTANT (it does not
//!   depend on a free coordinate) or an affine EXPRESSION of lane values;
//! * constants are computed once per clause (rows cached across clauses);
//! * a `Linear` row with free inputs becomes a lane `Linear` over just those
//!   inputs, the constant inputs folded into an interval bias;
//! * elementwise ops, masks, `ReduceSum`, `Div` by a constant, `Concat`, `Slice`
//!   and reshapes fuse into [`Op::Sparse`] terms (with coefficient-error bounds);
//! * a trailing `Sigmoid` (and `const - Sigmoid`, the mscn dual) becomes a
//!   bound on the LOGIT, with directed rounding;
//! * `#lowdim-fold-sigdiff`: `±Sigmoid(zA) ± Sigmoid(zB) + const` (the mscn
//!   dual with BOTH copies reading free coordinates) makes the lane seed the
//!   2-wide `[zA, zB]`; each conjunct becomes two objectives whose rigorous
//!   lower bounds are pushed through an outward-rounded sigmoid and summed
//!   (`Combine::SigmoidSum`, `sig_sum_target`).
//!
//! # Soundness
//!
//! Constants are rigorous enclosures (outward rounding, Higham slack on every
//! dot product); coefficients carry explicit error bounds; the lane itself is
//! the audited forward-linear/backward-CROWN machinery. Mapping errors (a wrong
//! axis, a wrong broadcast) are what this module could get wrong, so admission
//! CROSS-CHECKS the folded program against ny's own graph forward at the box
//! centre and declines on any disagreement.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ny_tensor::BoundedTensor;

use super::{
    gamma_n, gslack, linear_parts, next_down, next_up, LowDimConfig, LowDimOutcome, Op, Program,
    SparseTerm, MAX_INPUT_DIM, MAX_TOTAL_NEURONS,
};

/// A sum of a few rounded non-negative terms, closed upward: the relative
/// factor dominates the few-ulp error of the component products and sums,
/// which a single `next_up` alone does not.
fn up(x: f64) -> f64 {
    next_up(x * (1.0 + 1e-12))
}
use crate::layers::Layer;
use crate::network::{GraphNetwork, NETWORK_INPUT};

/// Rows of constant `Linear` outputs cached across clauses (a key is the
/// FULL input row, never a hash: a collision would be unsound).
const ROW_CACHE_CAP: usize = 4096;

/// Shared across the clauses of one property.
#[derive(Default)]
pub struct FoldCache {
    shapes: Mutex<Option<Arc<HashMap<String, Vec<usize>>>>>,
    weights: Mutex<HashMap<(String, Vec<u32>), Arc<LinParts>>>,
    /// Constant-row folds keyed by a fast 64-bit hash of (node, row bits). The
    /// full row is verified OUTSIDE the lock on every hit; a hash collision just
    /// recomputes (never caches), so a hit is always the exact row's value.
    rows: Mutex<HashMap<u64, Arc<RowEntry>>>,
    /// Per `Linear` node: its weight (row-major `n_out x n_in`) and bias in f64.
    dense: Mutex<HashMap<String, Arc<(Vec<f64>, Vec<f64>)>>>,
}

impl FoldCache {
    /// A fresh cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// #lowdim-fold-rowhash: `compute()` for this (node, row), memoised.
    ///
    /// Measured (pin ny_fspeed_2f8da6264, cardinality_0_1000_2048): preparing 2000
    /// clauses took 19.4 s wall even in parallel. The old key was the full row's
    /// bit pattern (up to thousands of `u64`), SipHashed while holding the one
    /// global mutex, ~400 lookups per clause — serialised by the lock. Here the
    /// hash is a cheap word mix computed before locking, the lock covers only a
    /// u64 map probe, and the entry's full row is compared bit-for-bit after
    /// the lock is released.
    fn row_value(
        &self,
        name: &str,
        row: &[(f64, f64)],
        compute: impl FnOnce() -> Vec<(f64, f64)>,
    ) -> Arc<[(f64, f64)]> {
        let h = row_hash(name, row);
        let hit = self.rows.lock().expect("fold cache").get(&h).cloned();
        if let Some(e) = hit {
            if e.name == name
                && e.row.len() == row.len()
                && e.row
                    .iter()
                    .zip(row)
                    .all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits())
            {
                ROW_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Arc::clone(&e.vals);
            }
            // Collision: compute, do not overwrite the resident entry.
            ROW_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Arc::from(compute());
        }
        ROW_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let vals: Arc<[(f64, f64)]> = Arc::from(compute());
        let mut rc = self.rows.lock().expect("fold cache");
        if rc.len() < ROW_CACHE_CAP {
            rc.entry(h).or_insert_with(|| {
                Arc::new(RowEntry {
                    name: name.to_string(),
                    row: row.to_vec(),
                    vals: Arc::clone(&vals),
                })
            });
        }
        vals
    }
}

/// One memoised constant-row fold (see [`FoldCache::row_value`]).
struct RowEntry {
    name: String,
    row: Vec<(f64, f64)>,
    vals: Arc<[(f64, f64)]>,
}

/// Per-clause preparation CPU time split (summed over threads), for the
/// `[lowdim-fold-family]` observable.
pub(crate) static FOLD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static CHECK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Row-cache hit / miss counters for the `[lowdim-fold-family]` observable.
pub(crate) static ROW_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static ROW_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A cheap, deterministic 64-bit mix of the node name and the row's bit
/// patterns (multiply-xorshift per word). Only a bucket index: equality is
/// always re-checked on the full row.
fn row_hash(name: &str, row: &[(f64, f64)]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (row.len() as u64);
    for b in name.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(K);
    }
    for &(l, u) in row {
        h = (h ^ l.to_bits()).wrapping_mul(K);
        h ^= h >> 32;
        h = (h ^ u.to_bits()).wrapping_mul(K);
        h ^= h >> 29;
    }
    h
}

struct LinParts {
    w: Arc<[f64]>,
    w_pos: Arc<[f64]>,
    w_neg: Arc<[f64]>,
    row_abs: Arc<[f64]>,
}

/// An affine expression of lane values: `sum coef*v + k`, `k in [k_lo, k_hi]`.
#[derive(Debug, Clone)]
struct Expr {
    terms: Vec<SparseTerm>,
    k_lo: f64,
    k_hi: f64,
}

/// One folded graph element.
#[derive(Debug, Clone)]
enum E {
    C(f64, f64),
    X(Expr),
    /// `sgn * sigmoid(inner) + [off_lo, off_hi]` — only legal on the way to
    /// the output.
    Sig {
        inner: Expr,
        sgn: f64,
        off_lo: f64,
        off_hi: f64,
    },
    /// `#lowdim-fold-sigdiff`: `a_sgn * sigmoid(a) + b_sgn * sigmoid(b) +
    /// [off_lo, off_hi]` (the mscn dual `Sub(Sigmoid(copy A), Sigmoid(copy B))`
    /// with BOTH copies reading free coordinates) — only legal on the way to
    /// the output.
    Sig2 {
        a: Expr,
        a_sgn: f64,
        b: Expr,
        b_sgn: f64,
        off_lo: f64,
        off_hi: f64,
    },
}

impl E {
    /// A sigmoid output form: nothing but the offset/sign algebra of `Add`,
    /// `Sub` and constants may consume it.
    fn is_sigmoid(&self) -> bool {
        matches!(self, E::Sig { .. } | E::Sig2 { .. })
    }
}

fn iadd(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (next_down(a.0 + b.0), next_up(a.1 + b.1))
}

fn imul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let p = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
    let lo = p.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (next_down(lo), next_up(hi))
}

/// `(mid, rad)` with `[lo, hi] ⊆ [mid - rad, mid + rad]`.
pub(super) fn mid_rad(lo: f64, hi: f64) -> (f64, f64) {
    let mid = 0.5 * lo + 0.5 * hi;
    (mid, next_up((hi - mid).max(mid - lo)))
}

impl Expr {
    fn var(src: usize, idx: usize) -> Self {
        Self {
            terms: vec![SparseTerm {
                src: src as u32,
                idx: idx as u32,
                coef: 1.0,
                err: 0.0,
            }],
            k_lo: 0.0,
            k_hi: 0.0,
        }
    }

    fn add_const(mut self, c: (f64, f64)) -> Self {
        let (lo, hi) = iadd((self.k_lo, self.k_hi), c);
        self.k_lo = lo;
        self.k_hi = hi;
        self
    }

    fn add(mut self, other: &Expr) -> Self {
        for t in &other.terms {
            if let Some(u) = self
                .terms
                .iter_mut()
                .find(|u| u.src == t.src && u.idx == t.idx)
            {
                let sum = u.coef + t.coef;
                u.err = up(u.err + t.err + sum.abs() * f64::EPSILON);
                u.coef = sum;
            } else {
                self.terms.push(*t);
            }
        }
        self.add_const((other.k_lo, other.k_hi))
    }

    fn neg(mut self) -> Self {
        for t in &mut self.terms {
            t.coef = -t.coef;
        }
        let (lo, hi) = (-self.k_hi, -self.k_lo);
        self.k_lo = lo;
        self.k_hi = hi;
        self
    }

    /// Multiply by a constant in `[c_lo, c_hi]`.
    fn scale(mut self, c: (f64, f64)) -> E {
        if c.0 == 0.0 && c.1 == 0.0 {
            return E::C(0.0, 0.0);
        }
        let (cm, cr) = mid_rad(c.0, c.1);
        for t in &mut self.terms {
            let coef = t.coef * cm;
            t.err = up(t.coef.abs() * cr + t.err * (cm.abs() + cr) + coef.abs() * f64::EPSILON);
            t.coef = coef;
        }
        let (lo, hi) = imul((self.k_lo, self.k_hi), c);
        self.k_lo = lo;
        self.k_hi = hi;
        E::X(self)
    }

    /// A single plain reference `1 * lane[src][idx] + 0`.
    fn plain_ref(&self) -> Option<(u32, u32)> {
        match self.terms.as_slice() {
            [t] if t.coef == 1.0 && t.err == 0.0 && self.k_lo == 0.0 && self.k_hi == 0.0 => {
                Some((t.src, t.idx))
            }
            _ => None,
        }
    }
}

fn sigmoid_bounds(lo: f64, hi: f64) -> (f64, f64) {
    let s = |x: f64| 1.0 / (1.0 + (-x).exp());
    let l = next_down(s(lo) * (1.0 - 1e-12)).max(0.0);
    let h = next_up(s(hi) * (1.0 + 1e-12)).min(1.0);
    (l, h)
}

/// `(L, U)` with `L <= logit(p) <= U`, for `p` in `(0, 1)`.
fn logit_bounds(p: f64) -> (f64, f64) {
    let v = p.ln() - (1.0 - p).ln();
    let margin = next_up(1e-12 * (1.0 + v.abs()) + 1e-15);
    (next_down(v - margin), next_up(v + margin))
}

/// Folded tensor of one graph node.
struct Tensor {
    shape: Vec<usize>,
    e: Vec<E>,
}

pub(super) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

pub(super) fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        idx[i] = flat % shape[i].max(1);
        flat /= shape[i].max(1);
    }
    idx
}

/// Right-aligned numpy broadcast of `in_shape` onto `out_shape`: the flat
/// input index for every flat output index.
pub(super) fn broadcast_map(in_shape: &[usize], out_shape: &[usize]) -> Result<Vec<usize>, String> {
    if in_shape.len() > out_shape.len() {
        // A leading run of 1s may be dropped.
        let extra = in_shape.len() - out_shape.len();
        if in_shape[..extra].iter().all(|&n| n == 1) {
            return broadcast_map(&in_shape[extra..], out_shape);
        }
        return Err(format!("cannot broadcast {in_shape:?} to {out_shape:?}"));
    }
    let off = out_shape.len() - in_shape.len();
    for (i, &n) in in_shape.iter().enumerate() {
        if n != 1 && n != out_shape[off + i] {
            return Err(format!("cannot broadcast {in_shape:?} to {out_shape:?}"));
        }
    }
    let st = strides(in_shape);
    let total: usize = out_shape.iter().product();
    Ok((0..total)
        .map(|f| {
            let o = unravel(f, out_shape);
            in_shape
                .iter()
                .enumerate()
                .map(|(i, &n)| if n == 1 { 0 } else { o[off + i] * st[i] })
                .sum()
        })
        .collect())
}

struct Builder<'a> {
    ops: Vec<Op>,
    sizes: Vec<usize>,
    cache: &'a FoldCache,
}

impl Builder<'_> {
    fn push(&mut self, op: Op, size: usize) -> usize {
        self.ops.push(op);
        self.sizes.push(size);
        self.ops.len() - 1
    }

    /// A lane node holding exactly `xs` (in order); reuses an existing node
    /// when `xs` is that whole node referenced in order.
    fn materialize(&mut self, xs: &[&Expr]) -> usize {
        if let Some((src, _)) = xs.first().and_then(|x| x.plain_ref()) {
            let whole = self.sizes[src as usize] == xs.len()
                && xs
                    .iter()
                    .enumerate()
                    .all(|(j, x)| x.plain_ref() == Some((src, j as u32)));
            if whole {
                return src as usize;
            }
        }
        let terms: Vec<Vec<SparseTerm>> = xs.iter().map(|x| x.terms.clone()).collect();
        let mut reads: HashMap<(u32, u32), usize> = HashMap::new();
        for t in terms.iter().flatten() {
            *reads.entry((t.src, t.idx)).or_default() += 1;
        }
        let fanin = reads.values().copied().max().unwrap_or(1);
        let op = Op::Sparse {
            terms,
            k_lo: xs.iter().map(|x| x.k_lo).collect(),
            k_hi: xs.iter().map(|x| x.k_hi).collect(),
            fanin,
        };
        self.push(op, xs.len())
    }

    fn lin_parts(
        &self,
        node: &str,
        w: &[f64],
        n_out: usize,
        n_in: usize,
        cols: &[usize],
    ) -> Arc<LinParts> {
        let key = (
            node.to_string(),
            cols.iter().map(|&c| c as u32).collect::<Vec<_>>(),
        );
        if let Some(p) = self.cache.weights.lock().expect("fold cache").get(&key) {
            return Arc::clone(p);
        }
        let k = cols.len();
        let mut sub = vec![0.0f64; n_out * k];
        for i in 0..n_out {
            for (jj, &c) in cols.iter().enumerate() {
                sub[i * k + jj] = w[i * n_in + c];
            }
        }
        let (w_pos, w_neg, row_abs) = linear_parts(&sub, n_out, k);
        let parts = Arc::new(LinParts {
            w: sub.into(),
            w_pos,
            w_neg,
            row_abs,
        });
        self.cache
            .weights
            .lock()
            .expect("fold cache")
            .insert(key, Arc::clone(&parts));
        parts
    }
}

/// Rigorous `W x + b` for an all-constant row `x` (midpoint-radius).
fn const_row(w: &[f64], b: &[f64], n_out: usize, n_in: usize, x: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mr: Vec<(f64, f64)> = x.iter().map(|&(l, h)| mid_rad(l, h)).collect();
    (0..n_out)
        .map(|i| {
            let row = &w[i * n_in..(i + 1) * n_in];
            let (mut v, mut mag, mut rad) = (b[i], b[i].abs(), 0.0f64);
            for (wj, &(c, r)) in row.iter().zip(&mr) {
                v += wj * c;
                mag += (wj * c).abs();
                rad += wj.abs() * r;
            }
            // `rad` is a rounded sum of `n_in` rounded products: inflate by gamma.
            let rad = up(rad * (1.0 + 2.0 * gamma_n(n_in + 2)));
            let e = up(gslack(n_in + 2, mag) + rad);
            (next_down(v - e), next_up(v + e))
        })
        .collect()
}

/// How the graph output relates to the lane seed node (index 0 of it; indices
/// 0 and 1 for [`OutForm::SigDiff`]).
#[derive(Debug, Clone)]
enum OutForm {
    Const(f64, f64),
    /// `Y = z`.
    Lin,
    /// `Y = sgn * sigmoid(z) + [off_lo, off_hi]`.
    Sig {
        sgn: f64,
        off_lo: f64,
        off_hi: f64,
    },
    /// `#lowdim-fold-sigdiff`: `Y = a_sgn * sigmoid(zA) + b_sgn * sigmoid(zB) + c`,
    /// `c` in `[c_lo, c_hi]`, `zA = seed[0]`, `zB = seed[1]` (the mscn dual has
    /// `a_sgn = +1`, `b_sgn = -1`).
    SigDiff {
        a_sgn: f64,
        b_sgn: f64,
        c_lo: f64,
        c_hi: f64,
    },
}

/// A clause compiled to a lane program over its free coordinates.
pub struct Folded {
    prog: Program,
    /// Flat input indices of the free coordinates, in lane-input order.
    pub free: Vec<usize>,
    out: OutForm,
}

fn shapes_of(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    cache: &FoldCache,
) -> Result<Arc<HashMap<String, Vec<usize>>>, String> {
    let mut slot = cache.shapes.lock().expect("fold cache");
    if let Some(s) = slot.as_ref() {
        return Ok(Arc::clone(s));
    }
    let detailed = graph
        .propagate_ibp_detailed(input, 0.0)
        .map_err(|e| format!("shape probe (IBP) failed: {e}"))?;
    let mut shapes: HashMap<String, Vec<usize>> = detailed
        .nodes
        .into_iter()
        .map(|n| (n.name, n.output_shape))
        .collect();
    shapes.insert(NETWORK_INPUT.to_string(), input.shape().to_vec());
    let shapes = Arc::new(shapes);
    *slot = Some(Arc::clone(&shapes));
    Ok(shapes)
}

/// Compile `graph` over `input` (any shape) into a lane program over the
/// coordinates with `hi > lo`.
pub fn fold_program(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    cache: &FoldCache,
) -> Result<Folded, String> {
    graph
        .ensure_proof_eligible()
        .map_err(|error| error.to_string())?;
    let lo: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    if lo.iter().chain(&hi).any(|v| !v.is_finite()) || lo.iter().zip(&hi).any(|(l, h)| l > h) {
        return Err("input box is not a finite, ordered box".into());
    }
    let free: Vec<usize> = (0..lo.len()).filter(|&i| hi[i] > lo[i]).collect();
    let d = free.len();
    if d == 0 || d > MAX_INPUT_DIM {
        return Err(format!(
            "{d} free coordinates (lane admits 1..={MAX_INPUT_DIM})"
        ));
    }
    let shapes = shapes_of(graph, input, cache)?;
    let shape_of = |name: &str| -> Result<Vec<usize>, String> {
        shapes
            .get(name)
            .cloned()
            .ok_or_else(|| format!("no shape recorded for `{name}`"))
    };
    let in_shape = input.shape().to_vec();
    if shape_of(NETWORK_INPUT)? != in_shape {
        return Err("input shape differs from the cached shape probe".into());
    }
    let mut b = Builder {
        ops: vec![Op::Input],
        sizes: vec![d],
        cache,
    };
    let mut slot = vec![usize::MAX; lo.len()];
    for (j, &f) in free.iter().enumerate() {
        slot[f] = j;
    }
    let mut vals: HashMap<String, Tensor> = HashMap::new();
    vals.insert(
        NETWORK_INPUT.to_string(),
        Tensor {
            shape: in_shape,
            e: (0..lo.len())
                .map(|i| {
                    if slot[i] == usize::MAX {
                        E::C(lo[i], hi[i])
                    } else {
                        E::X(Expr::var(0, slot[i]))
                    }
                })
                .collect(),
        },
    );
    let order = graph
        .topological_sort()
        .map_err(|e| format!("topological sort failed: {e}"))?;
    // Remaining consumers per tensor: a tensor is dropped after its last use
    // (a 2048-wide DAG would otherwise hold every intermediate per clause).
    let mut uses: HashMap<String, usize> = HashMap::new();
    for name in &order {
        if let Some(node) = graph.node(name) {
            for i in node.inputs() {
                *uses.entry(i.clone()).or_default() += 1;
            }
        }
    }
    *uses.entry(graph.output_name().to_string()).or_default() += 1;
    for name in &order {
        if name == NETWORK_INPUT {
            continue;
        }
        let node = graph
            .node(name)
            .ok_or_else(|| format!("node `{name}` missing"))?;
        let out_shape = shape_of(name)?;
        let n_out_total: usize = out_shape.iter().product();
        let ins = node.inputs();
        let get = |i: usize| -> Result<&Tensor, String> {
            let s = ins
                .get(i)
                .ok_or_else(|| format!("`{name}` lacks input {i}"))?;
            vals.get(s)
                .ok_or_else(|| format!("`{name}` input `{s}` not folded"))
        };
        let e: Vec<E> = match node.layer() {
            Layer::Reshape(_) | Layer::Flatten(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
                let t = get(0)?;
                if t.e.len() != n_out_total {
                    return Err(format!("reshape `{name}` changes the element count"));
                }
                t.e.clone()
            }
            Layer::Slice(s) => {
                let t = get(0)?;
                let (axis, start, end) = s
                    .resolved_range(&t.shape)
                    .map_err(|e| format!("Slice `{name}`: {e}"))?;
                let mut expect = t.shape.clone();
                expect[axis] = end - start;
                if expect.iter().product::<usize>() != n_out_total {
                    return Err(format!("Slice `{name}`: {expect:?} vs real {out_shape:?}"));
                }
                let st = strides(&t.shape);
                (0..n_out_total)
                    .map(|f| {
                        let mut o = unravel(f, &expect);
                        o[axis] += start;
                        t.e[o.iter().zip(&st).map(|(a, b)| a * b).sum::<usize>()].clone()
                    })
                    .collect()
            }
            Layer::Concat(_) => {
                let parts: Vec<&Tensor> = (0..ins.len()).map(get).collect::<Result<_, _>>()?;
                let rank = out_shape.len();
                let axis = (0..rank)
                    .filter(|&k| {
                        parts.iter().all(|p| {
                            p.shape.len() == rank
                                && (0..rank).all(|j| j == k || p.shape[j] == out_shape[j])
                        }) && parts.iter().map(|p| p.shape[k]).sum::<usize>() == out_shape[k]
                    })
                    .collect::<Vec<_>>();
                let [axis] = axis.as_slice() else {
                    return Err(format!("Concat `{name}`: axis not determined by shapes"));
                };
                let axis = *axis;
                (0..n_out_total)
                    .map(|f| {
                        let mut o = unravel(f, &out_shape);
                        let mut which = 0;
                        while o[axis] >= parts[which].shape[axis] {
                            o[axis] -= parts[which].shape[axis];
                            which += 1;
                        }
                        let st = strides(&parts[which].shape);
                        parts[which].e[o.iter().zip(&st).map(|(a, b)| a * b).sum::<usize>()].clone()
                    })
                    .collect()
            }
            Layer::Linear(l) => {
                let t = get(0)?;
                let w32 = l.weight();
                let (n_out, n_in) = (w32.nrows(), w32.ncols());
                if t.shape.last() != Some(&n_in) || out_shape.last() != Some(&n_out) {
                    return Err(format!("Linear `{name}` is not a last-axis map here"));
                }
                let rows = t.e.len() / n_in;
                if rows * n_out != n_out_total {
                    return Err(format!("Linear `{name}` row count mismatch"));
                }
                let cached = b.cache.dense.lock().expect("fold cache").get(name).cloned();
                let dense = match cached {
                    Some(d) => d,
                    None => {
                        let mut w = Vec::with_capacity(n_out * n_in);
                        for i in 0..n_out {
                            for j in 0..n_in {
                                w.push(f64::from(w32[(i, j)]));
                            }
                        }
                        let bias: Vec<f64> = match l.bias() {
                            Some(bv) => bv.iter().map(|&v| f64::from(v)).collect(),
                            None => vec![0.0; n_out],
                        };
                        if bias.len() != n_out || w.iter().chain(&bias).any(|v| !v.is_finite()) {
                            return Err(format!("Linear `{name}` has a bad weight or bias"));
                        }
                        let d = Arc::new((w, bias));
                        b.cache
                            .dense
                            .lock()
                            .expect("fold cache")
                            .insert(name.clone(), Arc::clone(&d));
                        d
                    }
                };
                let (w, bias) = (&dense.0, &dense.1);
                let mut out = Vec::with_capacity(n_out_total);
                for r in 0..rows {
                    let row = &t.e[r * n_in..(r + 1) * n_in];
                    let free_cols: Vec<usize> =
                        (0..n_in).filter(|&c| !matches!(row[c], E::C(..))).collect();
                    if row.iter().any(E::is_sigmoid) {
                        return Err(format!("Linear `{name}` consumes a sigmoid"));
                    }
                    let consts: Vec<(f64, f64)> = row
                        .iter()
                        .map(|x| match x {
                            E::C(l, h) => (*l, *h),
                            _ => (0.0, 0.0),
                        })
                        .collect();
                    if free_cols.is_empty() {
                        let vals_row = b
                            .cache
                            .row_value(name, &consts, || const_row(w, bias, n_out, n_in, &consts));
                        out.extend(vals_row.iter().map(|&(l, h)| E::C(l, h)));
                        continue;
                    }
                    // Constant columns fold into an interval bias.
                    let zeroed: Vec<(f64, f64)> = consts
                        .iter()
                        .enumerate()
                        .map(|(c, &v)| {
                            if free_cols.contains(&c) {
                                (0.0, 0.0)
                            } else {
                                v
                            }
                        })
                        .collect();
                    // #lowdim-fold-rowcache: the interval bias of the constant
                    // columns is `const_row` of the ZEROED row, the same pure
                    // function the fully-constant path caches, keyed the same
                    // way (full bit pattern). Clauses of one property usually
                    // differ only in their free coordinates, so this row
                    // repeats across clauses; measured, recomputing it per
                    // clause was 21 s of 34 s on cardinality_0_1000_2048.
                    let bias_iv = b
                        .cache
                        .row_value(name, &zeroed, || const_row(w, bias, n_out, n_in, &zeroed));
                    let xs: Vec<&Expr> = free_cols
                        .iter()
                        .map(|&c| match &row[c] {
                            E::X(x) => x,
                            _ => unreachable!("free column"),
                        })
                        .collect();
                    let src = b.materialize(&xs);
                    let parts = b.lin_parts(name, w, n_out, n_in, &free_cols);
                    let node_idx = b.push(
                        Op::Linear {
                            src,
                            n_in: free_cols.len(),
                            w_pos: Arc::clone(&parts.w_pos),
                            w_neg: Arc::clone(&parts.w_neg),
                            w: Arc::clone(&parts.w),
                            bias_lo: bias_iv.iter().map(|v| v.0).collect(),
                            bias_hi: bias_iv.iter().map(|v| v.1).collect(),
                            row_abs: Arc::clone(&parts.row_abs),
                        },
                        n_out,
                    );
                    out.extend((0..n_out).map(|j| E::X(Expr::var(node_idx, j))));
                }
                out
            }
            Layer::ReLU(_) => {
                let t = get(0)?;
                let xs: Vec<&Expr> =
                    t.e.iter()
                        .filter_map(|x| match x {
                            E::X(x) => Some(x),
                            _ => None,
                        })
                        .collect();
                if t.e.iter().any(E::is_sigmoid) {
                    return Err(format!("ReLU `{name}` consumes a sigmoid"));
                }
                let relu_node = if xs.is_empty() {
                    None
                } else {
                    let src = b.materialize(&xs);
                    let n = b.sizes[src];
                    Some(b.push(Op::Relu { src }, n))
                };
                let mut k = 0;
                t.e.iter()
                    .map(|x| match x {
                        E::C(l, h) => E::C(l.max(0.0), h.max(0.0)),
                        _ => {
                            let r = E::X(Expr::var(relu_node.expect("has x"), k));
                            k += 1;
                            r
                        }
                    })
                    .collect()
            }
            Layer::AddConstant(c) => {
                let t = get(0)?;
                unary_const(
                    &t.e,
                    &t.shape,
                    &out_shape,
                    &c.constant,
                    |x, k| add_ec(x, (k, k)),
                    name,
                )?
            }
            Layer::SubConstant(c) => {
                let t = get(0)?;
                let rev = c.reverse;
                unary_const(
                    &t.e,
                    &t.shape,
                    &out_shape,
                    &c.constant,
                    |x, k| {
                        if rev {
                            neg_e(x).and_then(|n| add_ec(n, (k, k)))
                        } else {
                            add_ec(x, (-k, -k))
                        }
                    },
                    name,
                )?
            }
            Layer::MulConstant(c) => {
                let t = get(0)?;
                unary_const(
                    &t.e,
                    &t.shape,
                    &out_shape,
                    &c.constant,
                    |x, k| mul_ec(x, (k, k)),
                    name,
                )?
            }
            Layer::DivConstant(c) => {
                let t = get(0)?;
                unary_const(
                    &t.e,
                    &t.shape,
                    &out_shape,
                    &c.constant,
                    |x, k| {
                        if k == 0.0 {
                            return Err("division by zero".into());
                        }
                        let r = (next_down(1.0 / k), next_up(1.0 / k));
                        mul_ec(x, (r.0.min(r.1), r.0.max(r.1)))
                    },
                    name,
                )?
            }
            Layer::Add(_) | Layer::Sub(_) | Layer::MulBinary(_) | Layer::Div(_) => {
                if ins.len() != 2 {
                    return Err(format!("binary `{name}` has {} inputs", ins.len()));
                }
                let (ta, tb) = (get(0)?, get(1)?);
                let ma = broadcast_map(&ta.shape, &out_shape)?;
                let mb = broadcast_map(&tb.shape, &out_shape)?;
                let layer = node.layer();
                (0..n_out_total)
                    .map(|f| {
                        let (xa, xb) = (ta.e[ma[f]].clone(), tb.e[mb[f]].clone());
                        match layer {
                            Layer::Add(_) => add_ee(xa, xb),
                            Layer::Sub(_) => neg_e(xb).and_then(|nb| add_ee(xa, nb)),
                            Layer::MulBinary(_) => match (xa, xb) {
                                (E::C(l1, h1), E::C(l2, h2)) => {
                                    let (l, h) = imul((l1, h1), (l2, h2));
                                    Ok(E::C(l, h))
                                }
                                (x, E::C(l, h)) | (E::C(l, h), x) => mul_ec(x, (l, h)),
                                _ => Err("a product of two free values".into()),
                            },
                            _ => match (xa, xb) {
                                (x, E::C(l, h)) if l > 0.0 || h < 0.0 => {
                                    let (r1, r2) = (1.0 / l, 1.0 / h);
                                    let rl = next_down(r1.min(r2));
                                    let rh = next_up(r1.max(r2));
                                    match x {
                                        E::C(a, c) => {
                                            let (lo, hi) = imul((a, c), (rl, rh));
                                            Ok(E::C(lo, hi))
                                        }
                                        other => mul_ec(other, (rl, rh)),
                                    }
                                }
                                _ => Err("division by a free or zero-straddling value".into()),
                            },
                        }
                    })
                    .collect::<Result<Vec<E>, String>>()
                    .map_err(|e| format!("`{name}`: {e}"))?
            }
            Layer::ReduceSum(r) => {
                let t = get(0)?;
                let axes = reduce_axes(&t.shape, &out_shape, &r.axes, r.keepdims)
                    .ok_or_else(|| format!("ReduceSum `{name}`: axes not determined"))?;
                let mut kept: Vec<usize> = t.shape.clone();
                for &a in &axes {
                    kept[a] = 1;
                }
                if kept.iter().product::<usize>() != n_out_total {
                    return Err(format!("ReduceSum `{name}` output size mismatch"));
                }
                let mut acc: Vec<Option<E>> = vec![None; n_out_total];
                let kst = strides(&kept);
                for f in 0..t.e.len() {
                    let mut o = unravel(f, &t.shape);
                    for &a in &axes {
                        o[a] = 0;
                    }
                    let of: usize = o.iter().zip(&kst).map(|(a, b)| a * b).sum();
                    let x = t.e[f].clone();
                    acc[of] = Some(match acc[of].take() {
                        None => x,
                        Some(prev) => {
                            add_ee(prev, x).map_err(|e| format!("ReduceSum `{name}`: {e}"))?
                        }
                    });
                }
                acc.into_iter()
                    .map(|x| x.unwrap_or(E::C(0.0, 0.0)))
                    .collect()
            }
            Layer::Sigmoid(_) => {
                let t = get(0)?;
                t.e.iter()
                    .map(|x| match x {
                        E::C(l, h) => {
                            let (a, c) = sigmoid_bounds(*l, *h);
                            Ok(E::C(a, c))
                        }
                        E::X(x) => Ok(E::Sig {
                            inner: x.clone(),
                            sgn: 1.0,
                            off_lo: 0.0,
                            off_hi: 0.0,
                        }),
                        E::Sig { .. } | E::Sig2 { .. } => {
                            Err(format!("Sigmoid `{name}` of a sigmoid"))
                        }
                    })
                    .collect::<Result<Vec<E>, String>>()?
            }
            other => {
                return Err(format!(
                    "node `{name}` is {} (outside the fold fragment)",
                    other.layer_type()
                ))
            }
        };
        if e.len() != n_out_total {
            return Err(format!(
                "`{name}` folded to {} elements, real {out_shape:?}",
                e.len()
            ));
        }
        for i in ins {
            if let Some(u) = uses.get_mut(i) {
                *u = u.saturating_sub(1);
                if *u == 0 {
                    vals.remove(i);
                }
            }
        }
        vals.insert(
            name.clone(),
            Tensor {
                shape: out_shape,
                e,
            },
        );
    }
    let out = vals
        .get(graph.output_name())
        .ok_or("graph output was not folded")?;
    let [y] = out.e.as_slice() else {
        return Err(format!(
            "output has {} elements (the fold lane needs 1)",
            out.e.len()
        ));
    };
    // The seed node the objectives read: the output's affine pre-image (one
    // element), or for `#lowdim-fold-sigdiff` BOTH sigmoid pre-activations
    // `[zA, zB]` (two elements).
    let (seed_exprs, form): (Vec<&Expr>, OutForm) = match y {
        E::C(l, h) => (Vec::new(), OutForm::Const(*l, *h)),
        E::X(x) => (vec![x], OutForm::Lin),
        E::Sig {
            inner,
            sgn,
            off_lo,
            off_hi,
        } => (
            vec![inner],
            OutForm::Sig {
                sgn: *sgn,
                off_lo: *off_lo,
                off_hi: *off_hi,
            },
        ),
        E::Sig2 {
            a,
            a_sgn,
            b: bx,
            b_sgn,
            off_lo,
            off_hi,
        } => (
            vec![a, bx],
            OutForm::SigDiff {
                a_sgn: *a_sgn,
                b_sgn: *b_sgn,
                c_lo: *off_lo,
                c_hi: *off_hi,
            },
        ),
    };
    let seed = if seed_exprs.is_empty() {
        0
    } else {
        b.materialize(&seed_exprs)
    };
    let Builder { ops, sizes, .. } = b;
    let mut feeds_relu = vec![false; ops.len()];
    for op in &ops {
        if let Op::Relu { src } = op {
            feeds_relu[*src] = true;
        }
    }
    let total: usize = sizes.iter().sum();
    if total > MAX_TOTAL_NEURONS {
        return Err(format!(
            "{total} lane neurons exceed the cap {MAX_TOTAL_NEURONS}"
        ));
    }
    let mut prog = Program {
        ops,
        sizes,
        out: seed,
        d,
        const_terms: 4 * total + 64,
        total,
        seed,
        refine_max: 0,
        feeds_relu,
        mul_operand: Vec::new(),
        bilinear: false,
        sites: Vec::new(),
        site_of: Vec::new(),
        site_operand: Vec::new(),
        joint: false,
    };
    // #lsnc-lane: the structural indices (a folded program builds no product,
    // so it stays non-bilinear and siteless, bounding exactly as before).
    prog.derive_structure();
    Ok(Folded {
        prog,
        free,
        out: form,
    })
}

fn unary_const(
    xs: &[E],
    in_shape: &[usize],
    out_shape: &[usize],
    constant: &ndarray::ArrayD<f32>,
    f: impl Fn(E, f64) -> Result<E, String>,
    name: &str,
) -> Result<Vec<E>, String> {
    let n_out: usize = out_shape.iter().product();
    let mx = broadcast_map(in_shape, out_shape)?;
    let mk = broadcast_map(constant.shape(), out_shape)?;
    let k: Vec<f64> = constant.iter().map(|&v| f64::from(v)).collect();
    if k.iter().any(|v| !v.is_finite()) {
        return Err(format!("`{name}` has a non-finite constant"));
    }
    (0..n_out)
        .map(|j| f(xs[mx[j]].clone(), k[mk[j]]).map_err(|e| format!("`{name}`: {e}")))
        .collect()
}

fn neg_e(x: E) -> Result<E, String> {
    Ok(match x {
        E::C(l, h) => E::C(-h, -l),
        E::X(x) => E::X(x.neg()),
        E::Sig {
            inner,
            sgn,
            off_lo,
            off_hi,
        } => E::Sig {
            inner,
            sgn: -sgn,
            off_lo: -off_hi,
            off_hi: -off_lo,
        },
        E::Sig2 {
            a,
            a_sgn,
            b,
            b_sgn,
            off_lo,
            off_hi,
        } => E::Sig2 {
            a,
            a_sgn: -a_sgn,
            b,
            b_sgn: -b_sgn,
            off_lo: -off_hi,
            off_hi: -off_lo,
        },
    })
}

fn add_ec(x: E, c: (f64, f64)) -> Result<E, String> {
    Ok(match x {
        E::C(l, h) => {
            let (a, b) = iadd((l, h), c);
            E::C(a, b)
        }
        E::X(x) => E::X(x.add_const(c)),
        E::Sig {
            inner,
            sgn,
            off_lo,
            off_hi,
        } => {
            let (a, b) = iadd((off_lo, off_hi), c);
            E::Sig {
                inner,
                sgn,
                off_lo: a,
                off_hi: b,
            }
        }
        E::Sig2 {
            a,
            a_sgn,
            b,
            b_sgn,
            off_lo,
            off_hi,
        } => {
            let (lo, hi) = iadd((off_lo, off_hi), c);
            E::Sig2 {
                a,
                a_sgn,
                b,
                b_sgn,
                off_lo: lo,
                off_hi: hi,
            }
        }
    })
}

fn add_ee(a: E, b: E) -> Result<E, String> {
    match (a, b) {
        (E::C(l, h), other) | (other, E::C(l, h)) => add_ec(other, (l, h)),
        (E::X(x), E::X(y)) => Ok(E::X(x.add(&y))),
        // #lowdim-fold-sigdiff: two sigmoids of free values (the mscn dual's
        // `Sub(Sigmoid(copy A), Sigmoid(copy B))`, the `Sub` already folded into
        // the second one's sign). The offsets are exact interval sums.
        (
            E::Sig {
                inner: a,
                sgn: a_sgn,
                off_lo: la,
                off_hi: ha,
            },
            E::Sig {
                inner: b,
                sgn: b_sgn,
                off_lo: lb,
                off_hi: hb,
            },
        ) => {
            let (off_lo, off_hi) = iadd((la, ha), (lb, hb));
            Ok(E::Sig2 {
                a,
                a_sgn,
                b,
                b_sgn,
                off_lo,
                off_hi,
            })
        }
        (E::Sig2 { .. }, E::Sig { .. } | E::Sig2 { .. }) | (E::Sig { .. }, E::Sig2 { .. }) => {
            Err("more than two sigmoids combined".into())
        }
        _ => Err("a sigmoid combined with a free value".into()),
    }
}

fn mul_ec(x: E, c: (f64, f64)) -> Result<E, String> {
    match x {
        E::C(l, h) => {
            let (a, b) = imul((l, h), c);
            Ok(E::C(a, b))
        }
        E::X(x) => Ok(x.scale(c)),
        E::Sig { .. } | E::Sig2 { .. } => Err("a scaled sigmoid".into()),
    }
}

/// The reduced axes of `in_shape -> out_shape`, cross-checked against the
/// layer's own axes (as given, or shifted by a squeezed batch dimension).
pub(super) fn reduce_axes(
    in_shape: &[usize],
    out_shape: &[usize],
    axes: &[i64],
    keepdims: bool,
) -> Option<Vec<usize>> {
    let rank = in_shape.len() as i64;
    let candidates = [0i64, 1].into_iter().filter_map(|shift| {
        let mut set: Vec<usize> = axes
            .iter()
            .map(|&a| {
                let a = if a < 0 { a + rank } else { a - shift };
                (0..rank).contains(&a).then_some(a as usize)
            })
            .collect::<Option<Vec<_>>>()?;
        set.sort_unstable();
        set.dedup();
        let expect: Vec<usize> = if keepdims {
            in_shape
                .iter()
                .enumerate()
                .map(|(i, &n)| if set.contains(&i) { 1 } else { n })
                .collect()
        } else {
            in_shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !set.contains(i))
                .map(|(_, &n)| n)
                .collect()
        };
        (expect == out_shape).then_some(set)
    });
    let found: Vec<Vec<usize>> = candidates.collect();
    match found.as_slice() {
        [one] => Some(one.clone()),
        [a, b] if a == b => Some(a.clone()),
        _ => None,
    }
}

/// One conjunct of a clause: the clause is infeasible if `s * Y > t` holds on
/// the whole box, for ANY conjunct.
#[derive(Debug, Clone, Copy)]
pub struct FoldConjunct {
    /// +1 or -1.
    pub s: f64,
    pub t: f64,
}

/// What a conjunct becomes on the lane seed `z`.
#[derive(Debug)]
enum Target {
    /// Holds everywhere (the clause is infeasible).
    Always,
    /// Can never be established here.
    Never,
    /// Prove `c * z > thr`.
    Lane(f64, f64),
    /// `#lowdim-fold-sigdiff`: prove `sigmoid(wa * zA) + sigmoid(wb * zB) > thr`
    /// (`wa`, `wb` in `{+1, -1}`; objectives `[wa, 0]` and `[0, wb]` on the
    /// 2-wide seed). `shift`: at the interval constant's midpoint the conjunct's
    /// side is `s * Y = sigmoid(wa zA) + sigmoid(wb zB) + shift` (candidate test).
    SigSum {
        wa: f64,
        wb: f64,
        thr: f64,
        shift: f64,
    },
}

fn target(form: &OutForm, cj: FoldConjunct) -> Target {
    let (s, t) = (cj.s, cj.t);
    match *form {
        OutForm::Const(l, h) => {
            let holds = if s > 0.0 { l > t } else { -h > t };
            if holds {
                Target::Always
            } else {
                Target::Never
            }
        }
        OutForm::Lin => Target::Lane(s, t),
        OutForm::Sig {
            sgn,
            off_lo,
            off_hi,
        } => {
            // s*Y = a*sigmoid(z) + s*off, a = s*sgn; worst case s*off is its min.
            let a = s * sgn;
            let min_soff = if s > 0.0 { off_lo } else { -off_hi };
            let q = next_up(t - min_soff); // need a*sigmoid(z) > q
            if a > 0.0 {
                if q < 0.0 {
                    Target::Always
                } else if q >= 1.0 {
                    Target::Never
                } else {
                    let (_, u) = logit_bounds(q);
                    Target::Lane(1.0, u)
                }
            } else {
                let p = -q; // sigmoid(z) < p
                if p > 1.0 {
                    Target::Always
                } else if p <= 0.0 {
                    Target::Never
                } else {
                    let (l, _) = logit_bounds(p);
                    Target::Lane(-1.0, -l)
                }
            }
        }
        OutForm::SigDiff {
            a_sgn,
            b_sgn,
            c_lo,
            c_hi,
        } => sig_sum_target(s, t, a_sgn, b_sgn, c_lo, c_hi),
    }
}

/// `#lowdim-fold-sigdiff`: target `s * Y > t` for
/// `Y = a_sgn * sigmoid(zA) + b_sgn * sigmoid(zB) + c`, `c` in `[c_lo, c_hi]`.
///
/// With `al = s * a_sgn`, `be = s * b_sgn` (each `+1` or `-1`) and the exact
/// identity `-sigmoid(z) = sigmoid(-z) - 1`,
/// `al * sigmoid(zA) = sigmoid(al * zA) - ka` with `ka = (1 - al) / 2` in `{0, 1}`
/// (likewise `kb`), so
/// `s * Y = sigmoid(al zA) + sigmoid(be zB) - ka - kb + s * c`.
/// Its worst case over `c` is at `min(s * c)`, so `s * Y > t` holds wherever
/// `sigmoid(al zA) + sigmoid(be zB) > Q` with `Q >= t - min(s * c) + ka + kb`,
/// rounded UP. The sum lies in `(0, 2)`: `Q < 0` holds everywhere, `Q >= 2`
/// never can.
fn sig_sum_target(s: f64, t: f64, a_sgn: f64, b_sgn: f64, c_lo: f64, c_hi: f64) -> Target {
    // The identity above needs unit signs; anything else is not this form.
    if s.abs() != 1.0 || a_sgn.abs() != 1.0 || b_sgn.abs() != 1.0 {
        return Target::Never;
    }
    let (al, be) = (
        if s * a_sgn > 0.0 { 1.0 } else { -1.0 },
        if s * b_sgn > 0.0 { 1.0 } else { -1.0 },
    );
    let k = (if al > 0.0 { 0.0 } else { 1.0 }) + (if be > 0.0 { 0.0 } else { 1.0 });
    let min_sc = if s > 0.0 { c_lo } else { -c_hi };
    let q = next_up(t - min_sc);
    let thr = if k == 0.0 { q } else { next_up(q + k) };
    if !thr.is_finite() || thr >= 2.0 {
        Target::Never
    } else if thr < 0.0 {
        Target::Always
    } else {
        let c_mid = 0.5 * c_lo + 0.5 * c_hi;
        Target::SigSum {
            wa: al,
            wb: be,
            thr,
            shift: (if s > 0.0 { c_mid } else { -c_mid }) - k,
        }
    }
}

/// Plain f64 point value of the graph output through the folded program
/// (admission cross-check; never a verdict).
fn folded_point_output(f: &Folded, x_free: &[f64]) -> f64 {
    let z = || super::point_objective(&f.prog, x_free, &[1.0]);
    match f.out {
        OutForm::Const(l, h) => f64::midpoint(l, h),
        OutForm::Lin => z(),
        OutForm::Sig {
            sgn,
            off_lo,
            off_hi,
        } => sgn / (1.0 + (-z()).exp()) + f64::midpoint(off_lo, off_hi),
        // The FULL combined output: both sigmoids and the constant.
        OutForm::SigDiff {
            a_sgn,
            b_sgn,
            c_lo,
            c_hi,
        } => {
            let zs = super::point_values(&f.prog, x_free);
            match zs.as_slice() {
                [za, zb] => {
                    a_sgn / (1.0 + (-za).exp())
                        + b_sgn / (1.0 + (-zb).exp())
                        + f64::midpoint(c_lo, c_hi)
                }
                _ => f64::NAN,
            }
        }
    }
}

/// Compile, cross-check and target one clause: either a family clause ready
/// for the lane, or an immediate outcome (declined, or proved outright by a
/// conjunct that holds everywhere through the output transform).
fn prepare_clause(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    conjuncts: &[FoldConjunct],
    cache: &FoldCache,
) -> Result<super::FamClause, Box<LowDimOutcome>> {
    let (folded, mine) = fold_and_value(graph, input, cache)?;
    let t_check = Instant::now();
    // Admission cross-check against ny's own graph forward at the box centre.
    let centre = input.center();
    let theirs = BoundedTensor::concrete(centre)
        .map_err(|e| e.to_string())
        .and_then(|c| {
            graph
                .propagate_concrete_point(&c, None, None)
                .map_err(|e| e.to_string())
        })
        .ok()
        .and_then(|y| y.center().iter().next().map(|&v| f64::from(v)))
        .ok_or_else(|| Box::new(LowDimOutcome::Declined("cross-check forward failed".into())))?;
    CHECK_NS.fetch_add(
        t_check.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    cross_check(mine, theirs)?;
    finish_clause(folded, input, conjuncts)
}

/// Fold one clause and evaluate the folded program at the box centre.
fn fold_and_value(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    cache: &FoldCache,
) -> Result<(Folded, f64), Box<LowDimOutcome>> {
    let t_fold = Instant::now();
    let folded =
        fold_program(graph, input, cache).map_err(|e| Box::new(LowDimOutcome::Declined(e)))?;
    let centre = input.center();
    let flat: Vec<f32> = centre.iter().copied().collect();
    let x_free: Vec<f64> = folded
        .free
        .iter()
        .map(|&i| flat.get(i).map(|&v| f64::from(v)))
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| Box::new(LowDimOutcome::Declined("free index outside the box".into())))?;
    let mine = folded_point_output(&folded, &x_free);
    FOLD_NS.fetch_add(
        t_fold.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    Ok((folded, mine))
}

/// The admission cross-check: the folded program and ny's graph forward must
/// agree at the box centre (relative 1e-4). Not a proof — a guard against a
/// fold that compiled the wrong function.
fn cross_check(mine: f64, theirs: f64) -> Result<(), Box<LowDimOutcome>> {
    if !mine.is_finite() || (mine - theirs).abs() > 1e-4 * (1.0 + theirs.abs()) {
        return Err(Box::new(LowDimOutcome::Declined(format!(
            "fold cross-check failed: folded {mine:e} vs graph {theirs:e}"
        ))));
    }
    Ok(())
}

/// Graph forward at many box centres at once (one batched point forward per
/// chunk), for the per-clause cross-check. `None` for a chunk whose batched
/// forward failed; its clauses fall back to the single-point forward.
fn batched_graph_values(
    graph: &GraphNetwork,
    centres: &[ndarray::ArrayD<f32>],
) -> Option<Vec<f64>> {
    let views: Vec<ndarray::ArrayViewD<'_, f32>> = centres.iter().map(|c| c.view()).collect();
    let stacked = ndarray::stack(ndarray::Axis(0), &views).ok()?;
    let t = BoundedTensor::concrete(stacked).ok()?;
    let y = graph
        .propagate_concrete_point_preserve_leading_axis(&t, None)
        .ok()?
        .center();
    if y.ndim() == 0 || y.shape()[0] != centres.len() {
        return None;
    }
    (0..centres.len())
        .map(|b| {
            y.index_axis(ndarray::Axis(0), b)
                .iter()
                .next()
                .map(|&v| f64::from(v))
        })
        .collect()
}

/// Target the conjuncts through the folded output and build the lane clause.
fn finish_clause(
    folded: Folded,
    input: &BoundedTensor,
    conjuncts: &[FoldConjunct],
) -> Result<super::FamClause, Box<LowDimOutcome>> {
    let mut objs: Vec<Vec<f64>> = Vec::new();
    let mut thrs: Vec<f64> = Vec::new();
    let mut centre: Vec<(f64, f64)> = Vec::new();
    for &cj in conjuncts {
        match target(&folded.out, cj) {
            Target::Always => {
                return Err(Box::new(LowDimOutcome::Verified(
                    super::LowDimStats::default(),
                )))
            }
            Target::Never => {}
            Target::Lane(c, thr) => {
                objs.push(vec![c]);
                thrs.push(thr);
            }
            Target::SigSum { wa, wb, thr, shift } => {
                objs.push(vec![wa, 0.0]);
                objs.push(vec![0.0, wb]);
                thrs.push(thr);
                centre.push((cj.t, shift));
            }
        }
    }
    // A clause's conjuncts all come from ONE output form, so its kind is the
    // form's: only a SigDiff output yields `SigSum` targets.
    let combine = match folded.out {
        OutForm::SigDiff { .. } => super::Combine::SigmoidSum { centre },
        _ => super::Combine::Each,
    };
    if objs.is_empty() {
        return Err(Box::new(LowDimOutcome::Declined(
            "no conjunct is provable through the output".into(),
        )));
    }
    let pick = |t: &ndarray::ArrayD<f32>| -> Vec<f64> {
        folded
            .free
            .iter()
            .map(|&i| f64::from(*t.iter().nth(i).expect("free")))
            .collect()
    };
    Ok(super::FamClause {
        lo: pick(input.lower()),
        hi: pick(input.upper()),
        prog: folded.prog,
        objs,
        thrs,
        combine,
    })
}

/// Prove the clause over `input` infeasible: some conjunct `s * Y > t` holds
/// everywhere. Never claims a violation.
pub fn prove_clause_folded(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    conjuncts: &[FoldConjunct],
    deadline: Instant,
    cfg: &LowDimConfig,
    cache: &FoldCache,
) -> LowDimOutcome {
    match prepare_clause(graph, input, conjuncts, cache) {
        Err(outcome) => *outcome,
        Ok(clause) => super::prove_family_core(vec![clause], deadline, cfg)
            .pop()
            .unwrap_or_else(|| LowDimOutcome::Declined("empty family result".into())),
    }
}

/// Many clauses at once: prepared in parallel, grouped by program STRUCTURE
/// (clauses with the same free coordinates share the lane's weights and differ
/// only in constants), and each group run as ONE family so every wave reads a
/// shared weight matrix once for all its clauses. `inputs[i]`/`conjuncts[i]`
/// describe clause `i`; the result is one outcome per clause, in order.
pub fn prove_clauses_folded(
    graph: &GraphNetwork,
    inputs: &[BoundedTensor],
    conjuncts: &[Vec<FoldConjunct>],
    deadline: Instant,
    cfg: &LowDimConfig,
    cache: &FoldCache,
) -> Vec<LowDimOutcome> {
    use rayon::prelude::*;
    let n = inputs.len().min(conjuncts.len());
    let t_prep = Instant::now();
    // #lowdim-fold-batchcheck: fold every clause in parallel, then run the
    // per-clause cross-check as BATCHED graph forwards (chunks of CHECK_CHUNK
    // centres, chunks in parallel). Measured on ny_fspeed3_6add7cd84
    // (cardinality_0_1000_2048): the single-point check was 352.6 s of CPU for
    // 2000 clauses (a full forward through 2048x2048 layers each) against 25.7 s
    // of folding. Every clause is still checked; a chunk whose batched forward
    // fails falls back to the single-point forward for its clauses.
    const CHECK_CHUNK: usize = 64;
    let folded: Vec<Result<(Folded, f64), Box<LowDimOutcome>>> = (0..n)
        .into_par_iter()
        .map(|i| fold_and_value(graph, &inputs[i], cache))
        .collect();
    let t_check = Instant::now();
    let ok_idx: Vec<usize> = (0..n).filter(|&i| folded[i].is_ok()).collect();
    let theirs_by_chunk: Vec<Vec<Option<f64>>> = ok_idx
        .par_chunks(CHECK_CHUNK)
        .map(|chunk| {
            let centres: Vec<ndarray::ArrayD<f32>> =
                chunk.iter().map(|&i| inputs[i].center()).collect();
            match batched_graph_values(graph, &centres) {
                Some(v) => v.into_iter().map(Some).collect(),
                None => centres
                    .into_iter()
                    .map(|c| {
                        BoundedTensor::concrete(c)
                            .ok()
                            .and_then(|t| graph.propagate_concrete_point(&t, None, None).ok())
                            .and_then(|y| y.center().iter().next().map(|&v| f64::from(v)))
                    })
                    .collect(),
            }
        })
        .collect();
    CHECK_NS.fetch_add(t_check.elapsed().as_nanos() as u64, Relaxed);
    let mut theirs: Vec<Option<f64>> = vec![None; n];
    for (&i, v) in ok_idx.iter().zip(theirs_by_chunk.into_iter().flatten()) {
        theirs[i] = v;
    }
    let prepared: Vec<Result<super::FamClause, Box<LowDimOutcome>>> = folded
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            let (f, mine) = r?;
            let g = theirs[i].ok_or_else(|| {
                Box::new(LowDimOutcome::Declined("cross-check forward failed".into()))
            })?;
            cross_check(mine, g)?;
            finish_clause(f, &inputs[i], &conjuncts[i])
        })
        .collect();
    let prepare_secs = t_prep.elapsed().as_secs_f64();
    let mut out: Vec<Option<LowDimOutcome>> = vec![None; n];
    // Greedy structure groups: (representative clause index, member indices).
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut clauses: Vec<Option<super::FamClause>> = Vec::with_capacity(n);
    for (i, p) in prepared.into_iter().enumerate() {
        match p {
            Err(outcome) => {
                out[i] = Some(*outcome);
                clauses.push(None);
            }
            Ok(c) => {
                // Same structure AND same conjunct kind (a sigmoid-sum clause
                // never shares a family with a single-objective one).
                let g = groups.iter().position(|g| {
                    clauses[g[0]]
                        .as_ref()
                        .is_some_and(|r| super::same_family(r, &c))
                });
                clauses.push(Some(c));
                match g {
                    Some(g) => groups[g].push(i),
                    None => groups.push(vec![i]),
                }
            }
        }
    }
    // #lowdim-fold-batch's OWN observable: how the clauses grouped and where the
    // wall went. `families == prepared` means no two clauses shared a structure,
    // i.e. the batching never engaged and any cost difference is overhead.
    let prepared_n = clauses.iter().filter(|c| c.is_some()).count();
    // #lowdim-fold-sigdiff's OWN observable: how many prepared clauses are the
    // two-sigmoid kind, and how many of those the lane proved.
    let sigdiff_n = clauses
        .iter()
        .flatten()
        .filter(|c| matches!(c.combine, super::Combine::SigmoidSum { .. }))
        .count();
    let max_family = groups.iter().map(Vec::len).max().unwrap_or(0);
    let t_prove = Instant::now();
    let (mut domains, mut verified, mut families_run) = (0u64, 0usize, 0usize);
    let mut sigdiff_verified = 0usize;
    // #lowdim-bundle (4): armed, every family runs on ONE shared worklist
    // (`shared`); the per-clause accounting below is the same as the
    // sequential loop's. Unarmed, the sequential loop runs exactly as before.
    // M8a: `input_split.engine: bbe` also takes the shared branch
    // (`super::fold_shared`); the bounder there follows `batch_engine`.
    let shared_branch = super::fold_shared(cfg);
    if shared_branch {
        let fams: Vec<Vec<super::FamClause>> = groups
            .iter()
            .map(|members| {
                members
                    .iter()
                    .map(|&i| clauses[i].take().expect("grouped once"))
                    .collect()
            })
            .collect();
        let sigdiff_fam: Vec<bool> = fams
            .iter()
            .map(|f| {
                f.first()
                    .is_some_and(|c| matches!(c.combine, super::Combine::SigmoidSum { .. }))
            })
            .collect();
        if Instant::now() < deadline {
            families_run = groups.len();
        }
        let (results, shared_stats) = super::shared::prove_families_shared(fams, deadline, cfg);
        for ((members, res), &sigdiff) in groups.iter().zip(results).zip(&sigdiff_fam) {
            for (&i, r) in members.iter().zip(res) {
                match &r {
                    LowDimOutcome::Verified(st) => {
                        verified += 1;
                        sigdiff_verified += usize::from(sigdiff);
                        domains += st.domains;
                    }
                    LowDimOutcome::Unproved(st, _) => domains += st.domains,
                    LowDimOutcome::Declined(_) => {}
                }
                out[i] = Some(r);
            }
        }
        super::shared::emit_line(&shared_stats);
    }
    // M8a: the legacy family loop never consults the wave search. An armed
    // `lowdim_batch_engine: bbe` that reaches it says so (an armed-but-not-
    // arrived run is VOID, not negative).
    if !shared_branch && cfg.batch_engine == super::LowDimBatchEngine::Bbe && !groups.is_empty() {
        eprintln!(
            "[bbe] phase=input_split lane=lowdim scheduler=family_loop fallback=legacy \
             reason=family_loop"
        );
    }
    for members in groups.iter().filter(|_| !shared_branch) {
        // Behaviour is unchanged: a family reached after the deadline still runs
        // `prove_family_core`, which returns Unproved(Deadline) at once. Counted
        // so the observable says how many families had wall left when entered.
        if Instant::now() < deadline {
            families_run += 1;
        }
        let fam: Vec<super::FamClause> = members
            .iter()
            .map(|&i| clauses[i].take().expect("grouped once"))
            .collect();
        let sigdiff = fam
            .first()
            .is_some_and(|c| matches!(c.combine, super::Combine::SigmoidSum { .. }));
        let results = super::prove_family_core(fam, deadline, cfg);
        for (&i, r) in members.iter().zip(results) {
            match &r {
                LowDimOutcome::Verified(st) => {
                    verified += 1;
                    sigdiff_verified += usize::from(sigdiff);
                    domains += st.domains;
                }
                LowDimOutcome::Unproved(st, _) => domains += st.domains,
                LowDimOutcome::Declined(_) => {}
            }
            out[i] = Some(r);
        }
    }
    use std::sync::atomic::Ordering::Relaxed;
    eprintln!(
        "[lowdim-fold-family] clauses={n} prepared={prepared_n} families={} families_run={families_run} \
max_family={max_family} prepare_secs={prepare_secs:.3} prove_secs={:.3} domains={domains} verified={verified} \
fold_cpu_secs={:.3} check_cpu_secs={:.3} row_hits={} row_misses={} sigdiff={sigdiff_n} \
sigdiff_verified={sigdiff_verified}",
        groups.len(),
        t_prove.elapsed().as_secs_f64(),
        FOLD_NS.load(Relaxed) as f64 * 1e-9,
        CHECK_NS.load(Relaxed) as f64 * 1e-9,
        ROW_HITS.load(Relaxed),
        ROW_MISSES.load(Relaxed)
    );
    out.into_iter()
        .map(|o| o.unwrap_or_else(|| LowDimOutcome::Declined("clause not run".into())))
        .collect()
}

#[cfg(test)]
#[path = "fold_tests.rs"]
mod tests;
