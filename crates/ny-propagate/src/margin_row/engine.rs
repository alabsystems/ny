// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sparse margin-row CROWN backward over frozen gates (#twinwall).
//!
//! Port of the verified Python reference engine (validated against an
//! exact-rational falsifier harness during development),
//! functionally equivalent in `Parity` mode (selfchecked there vs
//! `core.crown_backward_rows` bit-identically, T1-T5), extended with a
//! CERTIFIED coefficient-error lane in `Outward` mode:
//!
//! * every coefficient matrix `L` is accompanied by an elementwise error
//!   matrix `E` with the invariant that the exact-real backward recursion
//!   applied to the *stored* upstream values stays within `L ± E`;
//! * ReLU sign-splits are exact (`|max(x,0)-max(y,0)| + |min(x,0)-min(y,0)|
//!   = |x-y|`), so `E` contracts by `max(alpha, s) <= 1` per gate and grows
//!   only by fresh rounding (`2u|L|` per elementwise step, Higham `gamma_n`
//!   per accumulation, `RHO_FOLD` per folded conv);
//! * concretization subtracts (lower) / adds (upper) the full error penalty
//!   `E . (|mid|+rad) + eb` plus a `gamma` envelope over the accumulation and
//!   rounds the final endpoint outward twice.
//!
//! Per-row single-neuron gate EXCEPTIONS reproduce the batched child-scoring
//! trick (engine.py T5: one pass scores all candidate children exactly).

use ndarray::Array2;
use ny_core::{NyError, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ops::Range;

use super::net::{axpy, conv_apply_backward, TwinNet, TwinOp};
use super::root::{LayerGates, RootGates};
use super::rounding::{certify_up, gamma_n, next_down, next_up, slack16, UNIT};

/// Backward direction of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneDir {
    /// Lower rows: concretize toward -inf.
    Lower,
    /// Upper rows: concretize toward +inf.
    Upper,
}

/// Per-layer dense gate override (piece-fixed trunk splits applied).
#[derive(Clone)]
pub struct GateVecs {
    /// Lower slopes.
    pub alpha: Vec<f64>,
    /// Upper slopes.
    pub s: Vec<f64>,
    /// Upper intercepts.
    pub c: Vec<f64>,
    /// `max(alpha, s)` (error contraction).
    pub ms: Vec<f64>,
}

/// #margin-row-beta: one trunk split's Lagrangian term for a backward pass.
///
/// For a domain whose region is `box ∩ {s_j * z_j(x) >= 0}` (per split), weak
/// duality gives, for ANY `beta_j >= 0`,
///
/// ```text
///   min_{region} f  >=  min_{region} [f - Σ_j beta_j * s_j * z_j]
///                  >=  min_{box}    relax(f - Σ_j beta_j * s_j * z_j)
/// ```
///
/// because `beta_j * s_j * z_j >= 0` on the region (so subtracting it never
/// raises the objective there), and the lane's relaxed backward pass +
/// concretize is a valid lower bound of any seeded functional over any set
/// containing the region. Symmetrically for the Upper lane with `+ Σ beta s z`.
/// A WRONG `beta` therefore costs tightness, never validity — the only
/// obligations are `beta >= 0` and the coefficient/error algebra at the
/// application site (see `apply_beta_terms`).
#[derive(Debug, Clone, Copy)]
pub struct BetaSplit {
    /// Absolute neuron index within the layer.
    pub neuron: usize,
    /// Split sign: `+1` = active branch (`z >= 0`), `-1` = inactive (`z <= 0`).
    pub sign: i8,
    /// Multiplier, must be finite and `> 0`. Callers filter zero terms: the
    /// engine also skips them so that a `beta = 0` entry cannot perturb a
    /// `-0.0` coefficient bit-wise.
    pub beta: f64,
}

/// #margin-row-beta-percol: one COLUMN's split-Lagrangian terms at a layer.
///
/// The per-column extension of [`BetaSplit`]: a domain carries one margin
/// objective per seed column, and each column's Lagrangian is independent —
/// weak duality holds PER COLUMN for any `beta >= 0` vector attached to that
/// column alone. The engine realizes a per-column term with the SAME certified
/// `apply_beta_terms` arithmetic, restricted to `col..col + 1` (exactly the
/// column-range shape the domain-stacked arm already exercises per block).
#[derive(Debug, Clone)]
pub struct PcBetaCol {
    /// Seed column (margin objective) these terms price.
    pub col: usize,
    /// The column's Lagrangian terms at this layer.
    pub terms: Vec<BetaSplit>,
}

/// Domain gate overrides keyed by trunk layer index.
#[derive(Default, Clone)]
pub struct DomainGates {
    /// Overridden layers.
    pub layers: BTreeMap<usize, GateVecs>,
    /// #margin-row-beta (`NY_MARGIN_ROW_BETA=1`): split-Lagrangian terms per
    /// trunk layer. Empty by default, in which case every pass is BIT-IDENTICAL
    /// to the pre-beta engine (the application site is skipped entirely).
    pub beta: BTreeMap<usize, Vec<BetaSplit>>,
    /// #margin-row-beta-percol (`NY_MARGIN_ROW_BETA_PERCOL=1`): PER-COLUMN
    /// split-Lagrangian terms per trunk layer. Applied AFTER `beta` (the two
    /// compose additively: a column's effective multiplier is the shared term
    /// plus its own — a sum of `beta >= 0` is a valid multiplier). Column
    /// indices are the SEED columns of the pass this gate set is built for;
    /// callers must never reuse a `beta_pc`-carrying dom on a pass with a
    /// different column layout (the candidate-score passes strip it). Empty by
    /// default ⇒ bit-identical passes.
    pub beta_pc: BTreeMap<usize, Vec<PcBetaCol>>,
    /// #clip-and-verify, the VERIFY half: this domain's region is EMPTY.
    ///
    /// Set when constrained re-concretization produces CROSSED bounds on any
    /// neuron: `l_new` under-estimates the constrained minimum and `u_new`
    /// over-estimates the constrained maximum (both pay their certified slack
    /// in the safe direction), so `l_new > u_new` forces `min > max` over
    /// `box ∩ halfspaces` — a Farkas certificate that the set is empty. The
    /// domain's true region is a SUBSET of that set (each halfspace is a
    /// necessary condition of its split), so an empty intersection discharges
    /// the domain with no counterexample possible inside it. The certificate
    /// is conservative: slack makes it strictly harder to trigger, never
    /// falsely.
    pub infeasible: bool,
}

/// One domain's exclusive ownership of a contiguous column range in a
/// cross-domain backward pass.
///
/// `exceptions` use DOMAIN-LOCAL row numbers (`0..columns.len()`).  The
/// stacked entry point validates the ordered, gap-free partition and shifts
/// those rows into the combined matrix itself.  Keeping the shift here makes
/// it impossible for the caller to accidentally attach a candidate exception
/// to another domain's gate block.
pub(crate) struct RowDomainGateBlock<'g> {
    /// Exclusive combined-matrix columns owned by this domain.
    pub(crate) columns: Range<usize>,
    /// Piece-fixed trunk gates for exactly these columns.
    pub(crate) gates: &'g DomainGates,
    /// Per-candidate exceptions, numbered relative to `columns.start`.
    pub(crate) exceptions: &'g Exceptions,
}

/// One per-row gate exception (batched child scoring).
#[derive(Debug, Clone, Copy)]
pub struct Exc {
    /// Pass row the exception applies to.
    pub row: usize,
    /// Absolute neuron index within the layer.
    pub neuron: usize,
    /// Replacement lower slope.
    pub a2: f64,
    /// Replacement upper slope.
    pub s2: f64,
    /// Replacement intercept.
    pub c2: f64,
}

/// Exceptions grouped by trunk layer. At most one exception per (row, layer).
#[derive(Default)]
pub struct Exceptions {
    /// Layer index -> entries.
    pub by_layer: BTreeMap<usize, Vec<Exc>>,
}

/// Seed rows at the head pre-activation `y` (`(n_y, R)`, column r = row r).
pub struct Seed {
    /// Coefficients.
    pub s: Array2<f64>,
    /// Certified elementwise coefficient error (None = exact, e.g. identity).
    pub e: Option<Array2<f64>>,
}

/// What a backward pass collects at trunk relus (#epoch-bab).
#[derive(Default, Clone, Copy)]
pub struct Collect<'c> {
    /// Gather per-unstable |coef| sums (branching shortlist input).
    pub unst_abs: bool,
    /// Capture per-row coefficient vectors at each layer's RETAINED neurons
    /// (Tier-0 trunk-variant ranker input; nearest-mode values, ranker-only).
    pub rows: Option<&'c super::root::RetainedRows>,
    /// #alpha-opt: capture the PRE-transform incoming coefficients on the relu
    /// OUTPUT at every layer's UNSTABLE neurons (`(unst.len(), R)` per layer).
    ///
    /// GRADIENT-ONLY: these values never feed a verdict. They are the `v` of
    /// the gate transform `lv = vp*alpha + vn*s` below, captured so the alpha
    /// optimizer can read `vp = max(v, 0)` — the exact coefficient that
    /// multiplies `alpha_j` in this pass — without re-deriving the walk.
    pub unst_rows: bool,
}

/// Result of one backward pass: rows at the network input.
pub struct PassOut {
    /// Coefficients `(n_in, R)`.
    pub a: Array2<f64>,
    /// Certified elementwise error on `a` (Outward mode only).
    pub e: Option<Array2<f64>>,
    /// Bias per row.
    pub b: Vec<f64>,
    /// Certified bias error per row.
    pub eb: Vec<f64>,
    /// `collect_unst` output: layer -> per-unstable |coef| sums.
    pub coll: Option<BTreeMap<usize, Vec<f64>>>,
    /// Tier-0 capture: layer -> `(n_retained, R)` incoming coefficients on
    /// the relu OUTPUT at the layer's retained neurons (#epoch-bab).
    pub coll_rows: Option<BTreeMap<usize, Array2<f64>>>,
    /// #alpha-opt capture: layer -> `(unst.len(), R)` PRE-transform incoming
    /// coefficients at the layer's unstable neurons (gradient-only; see
    /// [`Collect::unst_rows`]). Row order follows `LayerGates::unst`.
    pub unst_rows: Option<BTreeMap<usize, Array2<f64>>>,
}

/// The engine: borrows the compiled net + frozen root gates.
pub struct BackwardEngine<'a> {
    /// Compiled net.
    pub net: &'a TwinNet,
    /// Frozen root gates + box.
    pub root: &'a RootGates,
}

struct LaneMat {
    l: Array2<f64>,
    e: Option<Array2<f64>>,
}

impl<'a> BackwardEngine<'a> {
    /// New engine over a net + its root gates.
    pub fn new(net: &'a TwinNet, root: &'a RootGates) -> Self {
        Self { net, root }
    }

    fn outward(&self) -> bool {
        self.root.mode.outward()
    }

    /// Identity seed (for the 2 x n_y y-row refresh passes).
    pub fn identity_seed(&self) -> Seed {
        let n_y = self.net.n_y;
        let mut s = Array2::<f64>::zeros((n_y, n_y));
        for j in 0..n_y {
            s[[j, j]] = 1.0;
        }
        Seed { s, e: None }
    }

    /// One backward pass. `dom`: piece-fixed gate overrides; `exc`: per-row
    /// single-neuron exceptions; `collect_unst`: gather |coef| mass at
    /// unstable trunk neurons (branching shortlist input).
    pub fn run(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect_unst: bool,
    ) -> Result<PassOut> {
        self.run_collect(
            seed,
            dom,
            dir,
            exc,
            Collect {
                unst_abs: collect_unst,
                rows: None,
                unst_rows: false,
            },
        )
    }

    /// One backward pass with the full collection spec (#epoch-bab).
    pub fn run_collect(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect: Collect<'_>,
    ) -> Result<PassOut> {
        self.run_collect_impl(seed, dom, dir, exc, collect, None, None)
    }

    /// #backward-interm: one certified backward pass seeded at the INPUT of
    /// trunk relu op `relu_op` (identity-style rows on its pre-activation),
    /// walked through the already-frozen PREFIX gates and concretizable over
    /// the root box with the lane's unchanged `concretize_*`.
    ///
    /// SOUNDNESS: the prefix relus' gates are the same frozen `LayerGates`
    /// every verdict pass uses (valid DeepPoly relaxations on their certified
    /// boxes), the certified error lane starts at zero (identity seed is
    /// exact) and contracts/grows by the unchanged per-op algebra, so the
    /// concretized rows are valid outward bounds on this relu's pre-activation
    /// over the root box — an independent enclosure the caller may intersect
    /// (shrink-only) with the forward tableau's.
    pub(crate) fn run_prefix(&self, seed: &Seed, relu_op: usize, dir: LaneDir) -> Result<PassOut> {
        self.run_collect_impl(
            seed,
            None,
            dir,
            None,
            Collect::default(),
            None,
            Some(relu_op),
        )
    }

    /// Cross-domain row stack used only by the default-off margin-row
    /// candidate-score canary.
    ///
    /// Every block owns one ordered, contiguous, non-empty range and the
    /// ranges must exactly partition all seed columns.  Exceptions are local
    /// to their block and are shifted only after validation.  Any malformed
    /// layout fails closed before entering verdict arithmetic.
    pub(crate) fn run_domain_stacked(
        &self,
        seed: &Seed,
        blocks: &[RowDomainGateBlock<'_>],
        dir: LaneDir,
    ) -> Result<PassOut> {
        let r = seed.s.ncols();
        if seed
            .e
            .as_ref()
            .is_some_and(|e| e.raw_dim() != seed.s.raw_dim())
        {
            return Err(NyError::InvalidSpec(
                "margin_row: domain-stack seed/error shape mismatch".into(),
            ));
        }
        validate_domain_stack(self.root, blocks, r)?;
        let mut shifted = Exceptions::default();
        for block in blocks {
            for (&li, list) in &block.exceptions.by_layer {
                let dst = shifted.by_layer.entry(li).or_default();
                for x in list {
                    dst.push(Exc {
                        row: block.columns.start + x.row,
                        neuron: x.neuron,
                        a2: x.a2,
                        s2: x.s2,
                        c2: x.c2,
                    });
                }
            }
        }
        self.run_collect_impl(
            seed,
            None,
            dir,
            Some(&shifted),
            Collect {
                unst_abs: false,
                rows: None,
                unst_rows: false,
            },
            Some(blocks),
            None,
        )
    }

    fn run_collect_impl(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        exc: Option<&Exceptions>,
        collect: Collect<'_>,
        domain_stack: Option<&[RowDomainGateBlock<'_>]>,
        prefix_relu: Option<usize>,
    ) -> Result<PassOut> {
        let net = self.net;
        let outward = self.outward();
        let r = seed.s.ncols();
        let mut state: Vec<Option<LaneMat>> = (0..=net.ops.len()).map(|_| None).collect();
        let mut b = vec![0.0; r];
        let mut eb = vec![0.0; r];
        let limit = if let Some(pk) = prefix_relu {
            // #backward-interm PREFIX entry: seed rows directly at the INPUT
            // tensor of trunk relu op `pk` (its pre-activation `z`), walk only
            // the ops strictly before it. Every relu encountered has a smaller
            // layer index, so its gates are already frozen in `self.root` —
            // the same certified per-gate algebra as a head-seeded pass, on a
            // shorter walk. The seed here is exact (identity rows), so its
            // error lane starts at zero when the caller supplies none.
            if pk >= net.i_gemm1 {
                return Err(NyError::InvalidSpec(
                    "margin_row: prefix seed op is not in the trunk".into(),
                ));
            }
            let TwinOp::Relu { input, .. } = &net.ops[pk] else {
                return Err(NyError::InvalidSpec(
                    "margin_row: prefix seed op is not a relu".into(),
                ));
            };
            let n_pre = net.tsize[*input];
            if seed.s.nrows() != n_pre {
                return Err(NyError::shape_mismatch(vec![n_pre], vec![seed.s.nrows()]));
            }
            let e0 = if outward {
                Some(
                    seed.e
                        .clone()
                        .unwrap_or_else(|| Array2::<f64>::zeros(seed.s.raw_dim())),
                )
            } else {
                None
            };
            state[*input] = Some(LaneMat {
                l: seed.s.clone(),
                e: e0,
            });
            pk
        } else {
            if seed.s.nrows() != net.n_y {
                return Err(NyError::shape_mismatch(vec![net.n_y], vec![seed.s.nrows()]));
            }
            let (w1, b1, (n_y, n_h)) = net.gemm1();

            // --- Seed through gemm1: L0[k] = sum_j W1[j,k] * S[j]; b = S @ b1.
            let mut l0 = Array2::<f64>::zeros((n_h, r));
            {
                let ss = seed.s.as_slice().expect("standard layout");
                let dst = l0.as_slice_mut().expect("standard layout");
                dst.par_chunks_mut(r).enumerate().for_each(|(k, acc)| {
                    for j in 0..n_y {
                        let w = w1[j * n_h + k];
                        if w != 0.0 {
                            axpy(acc, w, &ss[j * r..(j + 1) * r]);
                        }
                    }
                });
            }
            for j in 0..n_y {
                let bj = b1[j];
                if bj != 0.0 {
                    axpy(&mut b, bj, row(&seed.s, j));
                }
            }
            let e0 = if outward {
                // E0 = |W1|^T (Es + g(|S| + Es)), certified; eb likewise via |b1|.
                let g = gamma_n(n_y + 2);
                let comb = combined_err(&seed.s, seed.e.as_ref(), g);
                let mut e0 = Array2::<f64>::zeros((n_h, r));
                {
                    let cs = comb.as_slice().expect("standard layout");
                    let dst = e0.as_slice_mut().expect("standard layout");
                    let g2 = gamma_n(n_y + 8);
                    dst.par_chunks_mut(r).enumerate().for_each(|(k, acc)| {
                        for j in 0..n_y {
                            let w = w1[j * n_h + k].abs();
                            if w != 0.0 {
                                axpy(acc, w, &cs[j * r..(j + 1) * r]);
                            }
                        }
                        for v in acc.iter_mut() {
                            *v = certify_up(*v, g2);
                        }
                    });
                }
                {
                    let cs = comb.as_slice().expect("standard layout");
                    for j in 0..n_y {
                        let bj = b1[j].abs();
                        if bj != 0.0 {
                            axpy(&mut eb, bj, &cs[j * r..(j + 1) * r]);
                        }
                    }
                    let g2 = gamma_n(n_y + 8);
                    for v in &mut eb {
                        *v = certify_up(*v, g2);
                    }
                }
                Some(e0)
            } else {
                None
            };
            let g1_in = match &net.ops[net.i_gemm1] {
                TwinOp::Gemm { input, .. } => *input,
                _ => unreachable!("validated"),
            };
            state[g1_in] = Some(LaneMat { l: l0, e: e0 });
            net.i_gemm1
        };
        let mut coll: Option<BTreeMap<usize, Vec<f64>>> = collect.unst_abs.then(BTreeMap::new);
        let mut coll_rows: Option<BTreeMap<usize, Array2<f64>>> =
            collect.rows.map(|_| BTreeMap::new());
        let mut unst_rows: Option<BTreeMap<usize, Array2<f64>>> =
            collect.unst_rows.then(BTreeMap::new);
        let lower = matches!(dir, LaneDir::Lower);

        for k in (0..limit).rev() {
            if state[k + 1].is_none() {
                continue;
            }
            let cur = state[k + 1].take().expect("checked above");
            match &net.ops[k] {
                TwinOp::Flatten { input } => merge_into(&mut state, *input, cur, outward),
                TwinOp::ChannelAffine {
                    input,
                    scale,
                    shift,
                    scale_rel_err,
                    shift_err,
                } => {
                    // Backward through `y = s ⊙ x + t`: bias folds `Σ v_j t_j`
                    // (with certified parameter error), coefficients scale by
                    // `s` with the same error-lane algebra as a fixed gate.
                    let mut cur = cur;
                    {
                        let ls = cur.l.as_slice().expect("standard layout");
                        let es = cur.e.as_ref().map(|e| e.as_slice().expect("layout"));
                        let n = scale.len();
                        let gam_b = gamma_n(n + 2);
                        for j in 0..n {
                            let tj = shift[j];
                            let te = shift_err[j];
                            if tj == 0.0 && te == 0.0 {
                                continue;
                            }
                            let lr = &ls[j * r..(j + 1) * r];
                            if tj != 0.0 {
                                axpy(&mut b, tj, lr);
                            }
                            if let Some(es) = es {
                                let er = &es[j * r..(j + 1) * r];
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    *ebv += (er[ri] + gam_b * mag) * tj.abs() + (mag + er[ri]) * te;
                                    // Running-accumulator rounding carry
                                    // (mirrors the conv-bias arm).
                                    if tj != 0.0 {
                                        *ebv += 2.0 * UNIT * b[ri].abs();
                                    }
                                }
                            }
                        }
                        if outward {
                            let gc = gamma_n(8 * n + 16);
                            for v in &mut eb {
                                *v = certify_up(*v, gc);
                            }
                        }
                    }
                    {
                        let ls = cur.l.as_slice_mut().expect("standard layout");
                        let rel = *scale_rel_err;
                        match cur.e.as_mut() {
                            Some(em) => {
                                let es = em.as_slice_mut().expect("standard layout");
                                ls.par_chunks_mut(r)
                                    .zip(es.par_chunks_mut(r))
                                    .enumerate()
                                    .for_each(|(j, (lrow, erow))| {
                                        let sj = scale[j];
                                        let sa = sj.abs();
                                        for (lv, ev) in lrow.iter_mut().zip(erow.iter_mut()) {
                                            let v = *lv;
                                            let lnew = v * sj;
                                            *ev = slack16(
                                                (*ev + rel * v.abs() + 4.0 * UNIT * v.abs()) * sa
                                                    + (rel + 4.0 * UNIT) * lnew.abs(),
                                            );
                                            *lv = lnew;
                                        }
                                    });
                            }
                            None => {
                                ls.par_chunks_mut(r).enumerate().for_each(|(j, lrow)| {
                                    let sj = scale[j];
                                    for lv in lrow.iter_mut() {
                                        *lv *= sj;
                                    }
                                });
                            }
                        }
                    }
                    merge_into(&mut state, *input, cur, outward);
                }
                TwinOp::Add { lhs, rhs } => {
                    let copy = LaneMat {
                        l: cur.l.clone(),
                        e: cur.e.clone(),
                    };
                    merge_into(&mut state, *lhs, cur, outward);
                    merge_into(&mut state, *rhs, copy, outward);
                }
                TwinOp::Relu { input, layer } => {
                    let li = *layer;
                    let rec = &self.root.layers[li];
                    let mut cur = cur;
                    if let Some(cmap) = coll.as_mut() {
                        if !rec.unst.is_empty() {
                            let ls = cur.l.as_slice().expect("standard layout");
                            let sums: Vec<f64> = rec
                                .unst
                                .iter()
                                .map(|&idx| {
                                    ls[idx * r..(idx + 1) * r].iter().map(|v| v.abs()).sum()
                                })
                                .collect();
                            cmap.insert(li, sums);
                        }
                    }
                    // Tier-0 capture (#epoch-bab): incoming coefficients on
                    // the relu OUTPUT at this layer's retained neurons,
                    // BEFORE the gate transform. Nearest values, ranker-only.
                    if let (Some(rmap), Some(ret)) = (coll_rows.as_mut(), collect.rows) {
                        if let Some(lr) = ret.layers.get(li) {
                            if !lr.idx.is_empty() {
                                let ls = cur.l.as_slice().expect("standard layout");
                                let mut vmat = Array2::<f64>::zeros((lr.idx.len(), r));
                                let vs = vmat.as_slice_mut().expect("standard layout");
                                for (row_i, &j) in lr.idx.iter().enumerate() {
                                    vs[row_i * r..(row_i + 1) * r]
                                        .copy_from_slice(&ls[j * r..(j + 1) * r]);
                                }
                                rmap.insert(li, vmat);
                            }
                        }
                    }
                    // #alpha-opt capture: PRE-transform coefficients at this
                    // layer's unstable neurons. Same values, same placement as
                    // the Tier-0 capture above, but indexed by `rec.unst`
                    // (every unstable neuron, not the retained subset).
                    // Gradient-only: nothing verdict-bearing reads this.
                    if let Some(umap) = unst_rows.as_mut() {
                        if !rec.unst.is_empty() {
                            let ls = cur.l.as_slice().expect("standard layout");
                            let mut vmat = Array2::<f64>::zeros((rec.unst.len(), r));
                            let vs = vmat.as_slice_mut().expect("standard layout");
                            for (row_i, &j) in rec.unst.iter().enumerate() {
                                vs[row_i * r..(row_i + 1) * r]
                                    .copy_from_slice(&ls[j * r..(j + 1) * r]);
                            }
                            umap.insert(li, vmat);
                        }
                    }
                    // The established uniform-domain arm below is kept
                    // byte-for-byte as the default path.  The canary enters
                    // this separate arm only through `run_domain_stacked`.
                    if let Some(blocks) = domain_stack {
                        apply_domain_stacked_relu(
                            &mut cur, &mut b, &mut eb, rec, li, r, lower, outward, blocks, exc,
                        )?;
                        merge_into(&mut state, *input, cur, outward);
                        continue;
                    }
                    let over = dom.and_then(|d| d.layers.get(&li));
                    let (alpha, s, c, ms) = match over {
                        Some(gv) => (&gv.alpha[..], &gv.s[..], &gv.c[..], &gv.ms[..]),
                        None => (&rec.alpha[..], &rec.s[..], &rec.c[..], &rec.ms[..]),
                    };
                    // Snapshot originals for exceptions BEFORE the in-place
                    // transform.
                    let excs = exc.and_then(|e| e.by_layer.get(&li));
                    let snaps: Vec<(f64, f64)> = excs
                        .map(|list| {
                            list.iter()
                                .map(|x| {
                                    let orig = cur.l[[x.neuron, x.row]];
                                    let eorig =
                                        cur.e.as_ref().map_or(0.0, |em| em[[x.neuron, x.row]]);
                                    (orig, eorig)
                                })
                                .collect()
                        })
                        .unwrap_or_default();

                    // Bias accumulation (before transform): lower lane uses the
                    // negative part, upper lane the positive part, against c.
                    let gam_b = gamma_n(rec.n + 2);
                    {
                        let ls = cur.l.as_slice().expect("standard layout");
                        let es = cur.e.as_ref().map(|e| e.as_slice().expect("layout"));
                        for j in 0..rec.n {
                            let cj = c[j];
                            if cj == 0.0 {
                                continue;
                            }
                            let lr = &ls[j * r..(j + 1) * r];
                            for (ri, bv) in b.iter_mut().enumerate() {
                                let v = lr[ri];
                                let part = if lower { v.min(0.0) } else { v.max(0.0) };
                                *bv += part * cj;
                            }
                            if let Some(es) = es {
                                let er = &es[j * r..(j + 1) * r];
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    // SOUNDNESS (running-bias accumulation): the fl() rounding
                                    // of `*bv += part*cj` above is <= u*|b[ri]| on the ACCUMULATOR
                                    // magnitude (dominated by the gemm1 seed / earlier layers),
                                    // NOT the term magnitude gam_b*mag. Carry 2u*|b_running| per
                                    // step (mirrors merge_into's 2u*|result|). Soundness-loosening:
                                    // only widens eb, so the certified dj can only shrink.
                                    *ebv += (er[ri] + gam_b * mag) * cj + 2.0 * UNIT * b[ri].abs();
                                }
                            }
                        }
                        if outward {
                            let gc = gamma_n(4 * rec.n + 16);
                            for v in &mut eb {
                                *v = certify_up(*v, gc);
                            }
                        }
                    }
                    // In-place gate transform + error contraction.
                    {
                        let cols = r;
                        let ls = cur.l.as_slice_mut().expect("standard layout");
                        match cur.e.as_mut() {
                            Some(em) => {
                                let es = em.as_slice_mut().expect("standard layout");
                                ls.par_chunks_mut(cols)
                                    .zip(es.par_chunks_mut(cols))
                                    .enumerate()
                                    .for_each(|(j, (lrow, erow))| {
                                        let (aj, sj, mj) = (alpha[j], s[j], ms[j]);
                                        for (lv, ev) in lrow.iter_mut().zip(erow.iter_mut()) {
                                            let v = *lv;
                                            let vp = v.max(0.0);
                                            let vn = v - vp;
                                            *lv = if lower {
                                                vp * aj + vn * sj
                                            } else {
                                                vp * sj + vn * aj
                                            };
                                            *ev = slack16(
                                                (*ev + 4.0 * UNIT * v.abs()) * mj
                                                    + 4.0 * UNIT * lv.abs(),
                                            );
                                        }
                                    });
                            }
                            None => {
                                ls.par_chunks_mut(cols).enumerate().for_each(|(j, lrow)| {
                                    let (aj, sj) = (alpha[j], s[j]);
                                    for lv in lrow.iter_mut() {
                                        let v = *lv;
                                        let vp = v.max(0.0);
                                        let vn = v - vp;
                                        *lv = if lower {
                                            vp * aj + vn * sj
                                        } else {
                                            vp * sj + vn * aj
                                        };
                                    }
                                });
                            }
                        }
                    }
                    // Exceptions: exact per-(row, neuron) gate replacement.
                    if let Some(list) = excs {
                        for (x, &(orig, eorig)) in list.iter().zip(&snaps) {
                            let vp = orig.max(0.0);
                            let vn = orig - vp;
                            let (lnew, bpart) = if lower {
                                (vp * x.a2 + vn * x.s2, vn * (x.c2 - c[x.neuron]))
                            } else {
                                (vp * x.s2 + vn * x.a2, vp * (x.c2 - c[x.neuron]))
                            };
                            cur.l[[x.neuron, x.row]] = lnew;
                            b[x.row] += bpart;
                            if let Some(em) = cur.e.as_mut() {
                                let m2 = x.a2.abs().max(x.s2.abs());
                                em[[x.neuron, x.row]] = slack16(
                                    (eorig + 4.0 * UNIT * orig.abs()) * m2
                                        + 4.0 * UNIT * lnew.abs(),
                                );
                                eb[x.row] = slack16(
                                    eb[x.row]
                                        + (eorig + 4.0 * UNIT * orig.abs())
                                            * (x.c2 - c[x.neuron]).abs()
                                        + 4.0 * UNIT * bpart.abs(),
                                );
                            }
                        }
                    }
                    // #margin-row-beta: split-Lagrangian coefficient shift,
                    // applied LAST (after the gate transform and exceptions,
                    // which never target a neuron this domain already split —
                    // candidate selection excludes used splits). No-op on an
                    // empty term list, hence bit-identical when disarmed.
                    if let Some(terms) = dom.and_then(|d| d.beta.get(&li)) {
                        apply_beta_terms(&mut cur, terms, lower, 0..r);
                    }
                    // #margin-row-beta-percol: per-COLUMN Lagrangian terms —
                    // the same certified apply, restricted to the owning
                    // column (the domain-stacked arm's column-range anatomy on
                    // the uniform path). Applied after the shared terms; the
                    // two compose additively per entry, each charging its own
                    // rounding. An out-of-range column is skipped: dropping a
                    // valid `beta >= 0` term only loosens, never unsounds.
                    if let Some(pcs) = dom.and_then(|d| d.beta_pc.get(&li)) {
                        for pc in pcs {
                            if pc.col < r {
                                apply_beta_terms(&mut cur, &pc.terms, lower, pc.col..pc.col + 1);
                            }
                        }
                    }
                    merge_into(&mut state, *input, cur, outward);
                }
                TwinOp::Conv(cv) => {
                    let n_out = net.tsize[k + 1];
                    debug_assert_eq!(cur.l.nrows(), n_out);
                    let p = cv.oshape.1 * cv.oshape.2;
                    // Bias: b += Lo @ brep (+ certified fold/bias error).
                    {
                        let ls = cur.l.as_slice().expect("standard layout");
                        let es = cur.e.as_ref().map(|e| e.as_slice().expect("layout"));
                        let gam_b = gamma_n(n_out + 2);
                        for j in 0..n_out {
                            let ch = j / p;
                            let bias = cv.bias[ch];
                            let berr = cv.bias_err[ch];
                            if bias == 0.0 && berr == 0.0 {
                                continue;
                            }
                            let lr = &ls[j * r..(j + 1) * r];
                            if bias != 0.0 {
                                axpy(&mut b, bias, lr);
                            }
                            if let Some(es) = es {
                                let er = &es[j * r..(j + 1) * r];
                                let ab = bias.abs();
                                for (ri, ebv) in eb.iter_mut().enumerate() {
                                    let mag = lr[ri].abs();
                                    *ebv += (er[ri] + gam_b * mag) * ab + (mag + er[ri]) * berr;
                                    // SOUNDNESS: the `axpy(&mut b, bias, lr)` accumulation above
                                    // rounds to <= u*|b[ri]| on the running accumulator magnitude
                                    // (dominated by earlier layers), not the term magnitude used
                                    // in gam_b*mag. Carry 2u*|b_running| per accumulating step
                                    // (mirrors merge_into). Soundness-loosening: widens eb only.
                                    if bias != 0.0 {
                                        *ebv += 2.0 * UNIT * b[ri].abs();
                                    }
                                }
                            }
                        }
                        if outward {
                            let gc = gamma_n(8 * n_out + 16);
                            for v in &mut eb {
                                *v = certify_up(*v, gc);
                            }
                        }
                    }
                    let n_in_t = net.tsize[cv.input];
                    let mut lnew = Array2::<f64>::zeros((n_in_t, r));
                    conv_apply_backward(cv, &cur.l, &mut lnew, false);
                    let enew = if outward {
                        let g = next_up(
                            gamma_n(cv.k_bwd + 2)
                                + cv.weight_rel_err
                                + gamma_n(cv.k_bwd + 2) * cv.weight_rel_err,
                        );
                        let comb = combined_err(&cur.l, cur.e.as_ref(), g);
                        let mut en = Array2::<f64>::zeros((n_in_t, r));
                        conv_apply_backward(cv, &comb, &mut en, true);
                        let g2 = gamma_n(cv.k_bwd + 8);
                        en.par_mapv_inplace(|v| certify_up(v, g2));
                        Some(en)
                    } else {
                        None
                    };
                    merge_into(&mut state, cv.input, LaneMat { l: lnew, e: enew }, outward);
                }
                TwinOp::Gemm { .. } => {
                    return Err(NyError::InvalidSpec(
                        "margin_row: unexpected trunk gemm".into(),
                    ))
                }
            }
        }
        let fin = state[0]
            .take()
            .ok_or_else(|| NyError::InvalidSpec("margin_row: no input rows".into()))?;
        let out = PassOut {
            a: fin.l,
            e: fin.e,
            b,
            eb,
            coll,
            coll_rows,
            unst_rows,
        };
        // Fail-closed NaN/Inf firewall: verdict math must never see non-finite.
        if out.a.iter().any(|v| !v.is_finite())
            || out.b.iter().any(|v| !v.is_finite())
            || out
                .e
                .as_ref()
                .is_some_and(|e| e.iter().any(|v| !v.is_finite()))
            || out.eb.iter().any(|v| !v.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "margin_row: non-finite backward rows".into(),
            ));
        }
        Ok(out)
    }

    /// The 2 x n_y y-row refresh (lower + upper lanes, identity seeds).
    pub fn y_rows(&self, dom: Option<&DomainGates>) -> Result<(PassOut, PassOut)> {
        let seed = self.identity_seed();
        let al = self.run(&seed, dom, LaneDir::Lower, None, false)?;
        let au = self.run(&seed, dom, LaneDir::Upper, None, false)?;
        Ok((al, au))
    }

    /// One backward pass, GPU-authoritative when the dark seam admits it.
    ///
    /// Semantically IDENTICAL to `run(seed, dom, dir, None, false)`: the seam
    /// either returns a certified [`PassOut`] the caller concretizes with the
    /// lane's own unchanged `concretize_*`, or refuses — in which case this is
    /// exactly the CPU call, bit-for-bit. With `NY_MARGIN_ROW_GPU` unset the
    /// seam refuses before touching anything, so every existing bound is
    /// unchanged by construction.
    pub(crate) fn run_seamed(
        &self,
        seed: &Seed,
        dom: Option<&DomainGates>,
        dir: LaneDir,
        ctx: &super::gpu_seam::SeamCtx<'_>,
    ) -> Result<PassOut> {
        // #margin-row-beta: the seam predates the beta terms and would compute
        // WITHOUT them (a looser but still sound bound). No current caller
        // passes a beta-carrying dom here; refuse defensively so that can
        // never change silently. Per-column terms (`beta_pc`) are refused for
        // the same reason.
        if dom.is_some_and(|d| !d.beta.is_empty() || !d.beta_pc.is_empty()) {
            return self.run(seed, dom, dir, None, false);
        }
        match super::gpu_seam::run_pass(self, seed, dom, dir, ctx) {
            Ok(pass) => Ok(pass),
            Err(_refused) => self.run(seed, dom, dir, None, false),
        }
    }

    /// The 2 x n_y y-row refresh, GPU-authoritative when the seam admits it.
    ///
    /// The identity seed is exact in f32 and carries no certified error, so it
    /// needs no `y_abs`; and the device publishes BOTH lanes from ONE walk, so
    /// an admitted refresh costs one dispatch instead of two CPU passes.
    pub(crate) fn y_rows_seamed(
        &self,
        dom: Option<&DomainGates>,
        ctx: &super::gpu_seam::SeamCtx<'_>,
    ) -> Result<(PassOut, PassOut)> {
        let seed = self.identity_seed();
        // #margin-row-beta: see `run_seamed` — a beta-carrying dom never
        // enters the seam (defensive; the y-pack never carries beta today).
        if dom.is_some_and(|d| !d.beta.is_empty() || !d.beta_pc.is_empty()) {
            let al = self.run(&seed, dom, LaneDir::Lower, None, false)?;
            let au = self.run(&seed, dom, LaneDir::Upper, None, false)?;
            return Ok((al, au));
        }
        match super::gpu_seam::run_pass_pair(self, &seed, dom, ctx) {
            Ok(pair) => Ok(pair),
            Err(_refused) => {
                let al = self.run(&seed, dom, LaneDir::Lower, None, false)?;
                let au = self.run(&seed, dom, LaneDir::Upper, None, false)?;
                Ok((al, au))
            }
        }
    }

    /// Concretize lower rows over the root box: per row
    /// `A@mid + b - |A|@rad` (Python parity), minus the certified penalty and
    /// rounded toward -inf in Outward mode.
    pub fn concretize_lower(&self, out: &PassOut) -> Vec<f64> {
        self.concretize(out, true)
    }

    /// Concretize upper rows over the root box (toward +inf in Outward mode).
    pub fn concretize_upper(&self, out: &PassOut) -> Vec<f64> {
        self.concretize(out, false)
    }

    fn concretize(&self, out: &PassOut, lower: bool) -> Vec<f64> {
        let root = self.root;
        let n_in = root.mid.len();
        let r = out.a.ncols();
        let asl = out.a.as_slice().expect("standard layout");
        let esl = out.e.as_ref().map(|e| e.as_slice().expect("layout"));
        let mut v = vec![0.0; r];
        let mut rr = vec![0.0; r];
        let mut tabs = vec![0.0; r];
        let mut pen = vec![0.0; r];
        for i in 0..n_in {
            let m = root.mid[i];
            let rd = root.rad[i];
            let xa = root.xabs[i];
            let arow = &asl[i * r..(i + 1) * r];
            for (ri, av) in arow.iter().enumerate() {
                v[ri] += av * m;
                let aa = av.abs();
                rr[ri] += aa * rd;
                tabs[ri] += aa * xa;
            }
            if let Some(es) = esl {
                let erow = &es[i * r..(i + 1) * r];
                for (ri, ev) in erow.iter().enumerate() {
                    pen[ri] += ev * xa;
                }
            }
        }
        let gam = gamma_n(n_in + 16);
        (0..r)
            .map(|ri| {
                let base = v[ri] + out.b[ri];
                let raw = if lower { base - rr[ri] } else { base + rr[ri] };
                if !root.mode.outward() {
                    return raw;
                }
                let slack = next_up(
                    certify_up(pen[ri] + out.eb[ri], 1e-14)
                        + next_up(gam * (tabs[ri] + out.b[ri].abs() + pen[ri] + out.eb[ri])),
                );
                if lower {
                    next_down(next_down(raw - slack))
                } else {
                    next_up(next_up(raw + slack))
                }
            })
            .collect()
    }
}

fn validate_domain_stack(
    root: &RootGates,
    blocks: &[RowDomainGateBlock<'_>],
    columns: usize,
) -> Result<()> {
    if columns == 0 || blocks.is_empty() {
        return Err(NyError::InvalidSpec(
            "margin_row: empty domain-stack layout".into(),
        ));
    }
    let mut next = 0usize;
    for (bi, block) in blocks.iter().enumerate() {
        if block.columns.start != next
            || block.columns.start >= block.columns.end
            || block.columns.end > columns
        {
            return Err(NyError::InvalidSpec(format!(
                "margin_row: malformed domain-stack block {bi}: {:?}, expected start {next}, total {columns}",
                block.columns
            )));
        }
        let width = block.columns.end - block.columns.start;
        for (&li, gates) in &block.gates.layers {
            let Some(rec) = root.layers.get(li) else {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: domain-stack block {bi} references missing gate layer {li}"
                )));
            };
            if gates.alpha.len() != rec.n
                || gates.s.len() != rec.n
                || gates.c.len() != rec.n
                || gates.ms.len() != rec.n
            {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: domain-stack block {bi} gate shape mismatch at layer {li}"
                )));
            }
            if gates
                .alpha
                .iter()
                .chain(&gates.s)
                .chain(&gates.c)
                .chain(&gates.ms)
                .any(|v| !v.is_finite())
            {
                return Err(NyError::NumericalInstability(format!(
                    "margin_row: non-finite domain-stack gate in block {bi}, layer {li}"
                )));
            }
        }
        // #margin-row-beta: a stacked block's Lagrangian terms fail closed on
        // any shape or sign problem BEFORE entering verdict arithmetic (the
        // apply site would skip them silently; a malformed stack should not
        // degrade quietly).
        // #margin-row-beta-percol: a stacked block's column space is
        // BLOCK-LOCAL (candidate rows), while `beta_pc` columns index the
        // margin objectives of the domain's own eval pass. No caller builds
        // such a block (the eval clears `beta_pc` before the dom reaches any
        // scoring path); refuse rather than silently misapply a column.
        if !block.gates.beta_pc.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "margin_row: domain-stack block {bi} carries per-column beta \
(column spaces differ; unsupported)"
            )));
        }
        for (&li, terms) in &block.gates.beta {
            let Some(rec) = root.layers.get(li) else {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: domain-stack block {bi} beta references missing layer {li}"
                )));
            };
            for t in terms {
                if t.neuron >= rec.n || t.sign == 0 {
                    return Err(NyError::InvalidSpec(format!(
                        "margin_row: malformed domain-stack beta in block {bi}, layer {li}"
                    )));
                }
                if !(t.beta.is_finite() && t.beta >= 0.0) {
                    return Err(NyError::NumericalInstability(format!(
                        "margin_row: non-finite/negative domain-stack beta in block {bi}, layer {li}"
                    )));
                }
            }
        }
        for (&li, list) in &block.exceptions.by_layer {
            let Some(rec) = root.layers.get(li) else {
                return Err(NyError::InvalidSpec(format!(
                    "margin_row: domain-stack block {bi} exception references missing layer {li}"
                )));
            };
            let mut rows = std::collections::BTreeSet::new();
            for x in list {
                if x.row >= width || x.neuron >= rec.n {
                    return Err(NyError::InvalidSpec(format!(
                        "margin_row: domain-stack block {bi} exception escapes its owner"
                    )));
                }
                if !rows.insert(x.row) {
                    return Err(NyError::InvalidSpec(format!(
                        "margin_row: duplicate domain-stack exception in block {bi}, layer {li}, row {}",
                        x.row
                    )));
                }
                if !(x.a2.is_finite() && x.s2.is_finite() && x.c2.is_finite()) {
                    return Err(NyError::NumericalInstability(format!(
                        "margin_row: non-finite domain-stack exception in block {bi}, layer {li}"
                    )));
                }
            }
        }
        next = block.columns.end;
    }
    if next != columns {
        return Err(NyError::InvalidSpec(format!(
            "margin_row: incomplete domain-stack coverage: ended at {next}, total {columns}"
        )));
    }
    Ok(())
}

#[inline]
fn block_gates<'a>(
    block: &'a RowDomainGateBlock<'_>,
    li: usize,
    rec: &'a LayerGates,
) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
    match block.gates.layers.get(&li) {
        Some(gv) => (&gv.alpha, &gv.s, &gv.c, &gv.ms),
        None => (&rec.alpha, &rec.s, &rec.c, &rec.ms),
    }
}

/// Apply one ReLU backward step with per-column domain-gate ownership.
///
/// For each individual column this executes the same neuron order, scalar
/// operations, error updates, and exception correction as the established
/// uniform-domain arm.  Only independent columns from several domains share
/// the surrounding dispatch.
#[allow(clippy::too_many_arguments)]
fn apply_domain_stacked_relu(
    cur: &mut LaneMat,
    b: &mut [f64],
    eb: &mut [f64],
    rec: &LayerGates,
    li: usize,
    r: usize,
    lower: bool,
    outward: bool,
    blocks: &[RowDomainGateBlock<'_>],
    exc: Option<&Exceptions>,
) -> Result<()> {
    let excs = exc.and_then(|e| e.by_layer.get(&li));
    let snaps: Vec<(f64, f64)> = excs
        .map(|list| {
            list.iter()
                .map(|x| {
                    let orig = cur.l[[x.neuron, x.row]];
                    let eorig = cur.e.as_ref().map_or(0.0, |em| em[[x.neuron, x.row]]);
                    (orig, eorig)
                })
                .collect()
        })
        .unwrap_or_default();

    // Keep the established per-column reduction order: neuron j is still the
    // outer loop, and every column accumulates exactly one j term at a time.
    let gam_b = gamma_n(rec.n + 2);
    {
        let ls = cur.l.as_slice().expect("standard layout");
        let es = cur.e.as_ref().map(|e| e.as_slice().expect("layout"));
        for j in 0..rec.n {
            for block in blocks {
                let (_, _, c, _) = block_gates(block, li, rec);
                let cj = c[j];
                if cj == 0.0 {
                    continue;
                }
                let row0 = j * r;
                for ri in block.columns.clone() {
                    let v = ls[row0 + ri];
                    let part = if lower { v.min(0.0) } else { v.max(0.0) };
                    b[ri] += part * cj;
                }
                if let Some(es) = es {
                    for ri in block.columns.clone() {
                        let mag = ls[row0 + ri].abs();
                        eb[ri] += (es[row0 + ri] + gam_b * mag) * cj + 2.0 * UNIT * b[ri].abs();
                    }
                }
            }
        }
        if outward {
            let gc = gamma_n(4 * rec.n + 16);
            for v in eb.iter_mut() {
                *v = certify_up(*v, gc);
            }
        }
    }

    // Independent columns use the gates of exactly their owning block.  The
    // parallel grain remains one neuron row, as in the uniform-domain path.
    {
        let cols = r;
        let ls = cur.l.as_slice_mut().expect("standard layout");
        match cur.e.as_mut() {
            Some(em) => {
                let es = em.as_slice_mut().expect("standard layout");
                ls.par_chunks_mut(cols)
                    .zip(es.par_chunks_mut(cols))
                    .enumerate()
                    .for_each(|(j, (lrow, erow))| {
                        for block in blocks {
                            let (alpha, s, _, ms) = block_gates(block, li, rec);
                            let (aj, sj, mj) = (alpha[j], s[j], ms[j]);
                            for ri in block.columns.clone() {
                                let v = lrow[ri];
                                let vp = v.max(0.0);
                                let vn = v - vp;
                                lrow[ri] = if lower {
                                    vp * aj + vn * sj
                                } else {
                                    vp * sj + vn * aj
                                };
                                erow[ri] = slack16(
                                    (erow[ri] + 4.0 * UNIT * v.abs()) * mj
                                        + 4.0 * UNIT * lrow[ri].abs(),
                                );
                            }
                        }
                    });
            }
            None => {
                ls.par_chunks_mut(cols).enumerate().for_each(|(j, lrow)| {
                    for block in blocks {
                        let (alpha, s, _, _) = block_gates(block, li, rec);
                        let (aj, sj) = (alpha[j], s[j]);
                        for ri in block.columns.clone() {
                            let v = lrow[ri];
                            let vp = v.max(0.0);
                            let vn = v - vp;
                            lrow[ri] = if lower {
                                vp * aj + vn * sj
                            } else {
                                vp * sj + vn * aj
                            };
                        }
                    }
                });
            }
        }
    }

    // Exceptions retain their original owning domain's intercept.  Validation
    // above guarantees every shifted row lies in exactly one block.
    if let Some(list) = excs {
        for (x, &(orig, eorig)) in list.iter().zip(&snaps) {
            let Some(block) = blocks.iter().find(|block| block.columns.contains(&x.row)) else {
                return Err(NyError::InvalidSpec(
                    "margin_row: stacked exception has no domain owner".into(),
                ));
            };
            let (_, _, c, _) = block_gates(block, li, rec);
            let vp = orig.max(0.0);
            let vn = orig - vp;
            let (lnew, bpart) = if lower {
                (vp * x.a2 + vn * x.s2, vn * (x.c2 - c[x.neuron]))
            } else {
                (vp * x.s2 + vn * x.a2, vp * (x.c2 - c[x.neuron]))
            };
            cur.l[[x.neuron, x.row]] = lnew;
            b[x.row] += bpart;
            if let Some(em) = cur.e.as_mut() {
                let m2 = x.a2.abs().max(x.s2.abs());
                em[[x.neuron, x.row]] =
                    slack16((eorig + 4.0 * UNIT * orig.abs()) * m2 + 4.0 * UNIT * lnew.abs());
                eb[x.row] = slack16(
                    eb[x.row]
                        + (eorig + 4.0 * UNIT * orig.abs()) * (x.c2 - c[x.neuron]).abs()
                        + 4.0 * UNIT * bpart.abs(),
                );
            }
        }
    }
    // #margin-row-beta, stacked arm: each block's Lagrangian terms apply to
    // exactly its own columns — same math and same (post-exception) placement
    // as the uniform arm, so a stacked pass stays bit-identical to the
    // per-domain passes it replaces.
    for block in blocks {
        if let Some(terms) = block.gates.beta.get(&li) {
            apply_beta_terms(cur, terms, lower, block.columns.clone());
        }
    }
    Ok(())
}

/// #margin-row-beta: add each split's Lagrangian term to the coefficient the
/// backward pass now holds on that split neuron's PRE-activation `z`.
///
/// Placement: called at trunk relu `li` AFTER the gate transform, i.e. when
/// `cur.l[[neuron, .]]` is the coefficient multiplying `z_{li,neuron}` in the
/// relaxed functional. Adding `delta` there is exactly seeding the extra term
/// `delta * z_j(x)` through the remaining prefix walk:
///
/// * Lower lane bounds `f - Σ beta_j s_j z_j` from below ⇒ `delta = -s_j*beta_j`;
/// * Upper lane bounds `f + Σ beta_j s_j z_j` from above ⇒ `delta = +s_j*beta_j`.
///
/// See [`BetaSplit`] for the weak-duality argument that makes ANY `beta >= 0`
/// valid on the split region.
///
/// # Error-lane invariant (the load-bearing line)
///
/// The certified contraction `ms = max(alpha, s)` at this relu bounds the
/// Lipschitz constant of the GATE map `v -> vp*alpha + vn*s`, and that update
/// has already run. The beta shift is an ADDITIVE CONSTANT in the output
/// coefficient — `T_beta(v) = T(v) + delta` has the SAME Lipschitz constant as
/// `T` — so the carried upstream error needs NO rescaling (no `alpha_eff`
/// outside `[0,1]` ever enters the contraction). The only fresh error is the
/// one f64 addition below (`delta` itself is exact: `beta * (±1.0)`), bounded
/// by `u * |fl(l + delta)|` and charged at double weight. Loosening-only.
fn apply_beta_terms(cur: &mut LaneMat, terms: &[BetaSplit], lower: bool, cols: Range<usize>) {
    let r = cur.l.ncols();
    let n = cur.l.nrows();
    let ls = cur.l.as_slice_mut().expect("standard layout");
    let mut es = cur.e.as_mut().map(|e| e.as_slice_mut().expect("layout"));
    for t in terms {
        // Fail-safe skips: a non-positive or non-finite beta is refused here
        // (beta = 0 must not even perturb a `-0.0` coefficient bit-wise, and
        // a negative beta would flip the duality direction — unsound), as is
        // a signless term (delta would be ±0.0) or an out-of-range neuron.
        if !(t.beta.is_finite() && t.beta > 0.0) || t.sign == 0 || t.neuron >= n {
            continue;
        }
        let s = f64::from(t.sign.signum());
        let delta = if lower { -s * t.beta } else { s * t.beta };
        let row0 = t.neuron * r;
        for ri in cols.clone() {
            let lv = ls[row0 + ri] + delta;
            ls[row0 + ri] = lv;
            if let Some(es) = es.as_deref_mut() {
                es[row0 + ri] = slack16(es[row0 + ri] + 2.0 * UNIT * lv.abs());
            }
        }
    }
}

/// Build the per-layer dense gate override for a set of trunk piece-fixes
/// (`(layer, unst_pos, dir)`), mirroring `engine.py::domain_gates`.
pub fn domain_gates(root: &RootGates, trunk_splits: &[(usize, usize, i8)]) -> DomainGates {
    let mut dg = domain_gates_split_only(root, trunk_splits);
    clip_tighten_gates(root, trunk_splits, &mut dg);
    dg
}

/// #clip-and-verify: narrow OTHER unstable neurons using the halfspaces this
/// domain's splits imply, and rebuild their relaxation gates from the tightened
/// box.
///
/// This is what makes one split tighten the WHOLE domain. Tightening the output
/// y-box instead was measured inert (96 of 100 rows moved, trajectory
/// bit-identical) because nothing downstream consumes it; a narrowed
/// INTERMEDIATE `[l, u]` shrinks that neuron's ReLU triangle and so tightens
/// every backward pass through it.
///
/// # Soundness
///
/// `tighten_intermediate` only ever shrinks an interval, and the gates are
/// rebuilt by the SAME certified functions the root uses — `gates_from_box`
/// then `repair_upper_lines` in outward mode. A neuron that becomes stable
/// under the tightened box gets the exact fixed line, which is what the root
/// would have produced had it known the bound.
fn clip_tighten_gates(root: &RootGates, trunk_splits: &[(usize, usize, i8)], dg: &mut DomainGates) {
    if trunk_splits.is_empty() || !clip_interm_enabled() {
        return;
    }
    let hs = super::clip::halfspaces_for_splits(root, trunk_splits);
    if hs.is_empty() {
        return;
    }
    let (x0, eps) = (root.mid.as_slice(), root.rad.as_slice());
    let topk = clip_interm_topk();
    for (li, layer) in root.layers.iter().enumerate() {
        if layer.clip_rows.is_none() {
            continue;
        }
        let (tight, crossed) = super::clip::tighten_intermediate(layer, &hs, x0, eps, topk);
        if crossed > 0 {
            // VERIFY: crossed bounds certify the region empty. Stop tightening
            // — the caller discharges the whole domain.
            super::clip::report_infeasible(li, crossed);
            dg.infeasible = true;
            return;
        }
        if tight.is_empty() {
            continue;
        }
        let gv = dg.layers.entry(li).or_insert_with(|| GateVecs {
            alpha: layer.alpha.clone(),
            s: layer.s.clone(),
            c: layer.c.clone(),
            ms: layer.ms.clone(),
        });
        // Rebuild only the touched neurons, through the certified path.
        let (mut lv, mut uv) = (layer.l.clone(), layer.u.clone());
        let mut touched = Vec::with_capacity(tight.len());
        for t in &tight {
            // A neuron this domain SPLIT already carries an exact fixed line;
            // do not overwrite it with a relaxation.
            if trunk_splits
                .iter()
                .any(|&(l2, p2, _)| l2 == li && p2 == t.pos)
            {
                continue;
            }
            lv[t.idx] = t.l;
            uv[t.idx] = t.u;
            touched.push(t.idx);
        }
        if touched.is_empty() {
            continue;
        }
        let (alpha2, mut s2, mut c2) = super::root::gates_from_box(&lv, &uv);
        if root.mode.outward() {
            super::root::repair_upper_lines(&lv, &uv, &mut s2, &mut c2);
        }
        for &i in &touched {
            gv.alpha[i] = alpha2[i];
            gv.s[i] = s2[i];
            gv.c[i] = c2[i];
            // Root derives `ms = max(alpha, s)` (root.rs), and `s = u/(u-l)`
            // MOVES when the bounds tighten — so it must be re-derived, not
            // carried over. This also covers a neuron that became stable:
            // active gives (1,1) -> 1, inactive (0,0) -> 0.
            gv.ms[i] = alpha2[i].max(s2[i]);
        }
        super::clip::report_interm(li, touched.len(), tight.len());
    }
}

/// #clip-and-verify intermediate tightening: default OFF.
fn clip_interm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_INTERM)
            .value
            .as_bool()
    })
}

/// How many unstable neurons per layer to re-concretize. Matches
/// alpha-beta-CROWN's `bab.clip.interm_topk` default of 20.
fn clip_interm_topk() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_TOPK)
            .value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(20)
    })
}

fn domain_gates_split_only(root: &RootGates, trunk_splits: &[(usize, usize, i8)]) -> DomainGates {
    let mut dg = DomainGates::default();
    for &(li, pos, d) in trunk_splits {
        let rec = &root.layers[li];
        let idx = rec.unst[pos];
        let gv = dg.layers.entry(li).or_insert_with(|| GateVecs {
            alpha: rec.alpha.clone(),
            s: rec.s.clone(),
            c: rec.c.clone(),
            ms: rec.ms.clone(),
        });
        if d > 0 {
            gv.alpha[idx] = 1.0;
            gv.s[idx] = 1.0;
            gv.c[idx] = 0.0;
            gv.ms[idx] = 1.0;
        } else {
            gv.alpha[idx] = 0.0;
            gv.s[idx] = 0.0;
            gv.c[idx] = 0.0;
            gv.ms[idx] = 0.0;
        }
    }
    dg
}

fn row(m: &Array2<f64>, j: usize) -> &[f64] {
    let r = m.ncols();
    &m.as_slice().expect("standard layout")[j * r..(j + 1) * r]
}

/// `E + g * (|L| + E)` elementwise (error input to a linear step).
fn combined_err(l: &Array2<f64>, e: Option<&Array2<f64>>, g: f64) -> Array2<f64> {
    let mut comb = Array2::<f64>::zeros(l.raw_dim());
    let cols = l.ncols();
    let cs = comb.as_slice_mut().expect("standard layout");
    let ls = l.as_slice().expect("standard layout");
    match e {
        Some(em) => {
            let es = em.as_slice().expect("standard layout");
            cs.par_chunks_mut(cols)
                .zip(ls.par_chunks(cols).zip(es.par_chunks(cols)))
                .for_each(|(c, (lr, er))| {
                    for ((cv, &lv), &ev) in c.iter_mut().zip(lr).zip(er) {
                        *cv = ev + g * (lv.abs() + ev);
                    }
                });
        }
        None => {
            cs.par_chunks_mut(cols)
                .zip(ls.par_chunks(cols))
                .for_each(|(c, lr)| {
                    for (cv, &lv) in c.iter_mut().zip(lr) {
                        *cv = g * lv.abs();
                    }
                });
        }
    }
    comb
}

/// Accumulate a contribution into a tensor's lane state (elementwise add;
/// certified add-rounding into the error lane when outward).
fn merge_into(state: &mut [Option<LaneMat>], id: usize, contrib: LaneMat, outward: bool) {
    match state[id].take() {
        None => state[id] = Some(contrib),
        Some(mut acc) => {
            acc.l += &contrib.l;
            match (acc.e.as_mut(), contrib.e) {
                (Some(ea), Some(ecb)) => {
                    let cols = ea.ncols();
                    let es = ea.as_slice_mut().expect("standard layout");
                    let cs = ecb.as_slice().expect("standard layout");
                    let ls = acc.l.as_slice().expect("standard layout");
                    es.par_chunks_mut(cols)
                        .zip(cs.par_chunks(cols).zip(ls.par_chunks(cols)))
                        .for_each(|(er, (cr, lr))| {
                            for ((ev, &cv), &lv) in er.iter_mut().zip(cr).zip(lr) {
                                *ev = slack16(*ev + cv + 2.0 * UNIT * lv.abs());
                            }
                        });
                }
                (None, None) => {}
                _ => debug_assert!(!outward, "error lanes must agree"),
            }
            state[id] = Some(acc);
        }
    }
}
