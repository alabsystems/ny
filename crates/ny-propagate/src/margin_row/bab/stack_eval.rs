// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-stack — the domain-stacked frontier's shared MAIN eval pass (margin-row
//! throughput plan, build B5), its `[mr-stack]` receipt, and the tree
//! trajectory hash.
//!
//! # What changes
//!
//! With `margin_row.domain_stack` armed, `expand_domain_stacked_batch` used to
//! run each domain's whole evaluation independently — including its main
//! certified pass at R = nf columns (1-3 on cifar100 under column_retire) —
//! and stacked only the candidate-score pass. Now a frontier batch's
//! evaluations run in three stages:
//!
//! 1. [`MarginRowBab::eval_prepare`] per domain, rayon-parallel (gates, y-box,
//!    margin seed, inherited β / domain-alpha state — everything the unsplit
//!    `eval_with_pack` body computed before its pass);
//! 2. ONE cross-domain stacked main pass over every seeded domain's seed
//!    columns ([`MarginRowBab::eval_main_passes`], through
//!    `BackwardEngine::run_domain_stacked_split`), with a per-domain fallback;
//! 3. [`MarginRowBab::eval_finish`] per domain, rayon-parallel (per-column
//!    directs, via-y composition, the β and domain-alpha ascents with their own
//!    trial passes, column retirement).
//!
//! The B1 probe (`col-scaling-v2`, `scratchpad/mrb1/probe_ny_mrb1_0ad011c01_*`)
//! measured one stacked pass over K = 8 domains at 0.15-0.20x the CPU of K
//! separate `run_collect` passes, bit-identical, on resnet_medium,
//! resnet_large and a tinyimagenet net.
//!
//! # Why it cannot move a bound
//!
//! Each domain's block of the stacked pass is BIT-IDENTICAL to the pass the
//! domain would have run alone (see `run_domain_stacked_split` for the op-by-op
//! argument), and stages 1 and 3 are the unsplit body's own statements in its
//! own order. So every domain's `EvalOut` — bound, retirement, β / alpha state,
//! captures — is bit-identical, and the serial merge sees exactly the steps it
//! saw before. Pinned by `stacked_main_eval_bit_identical_to_per_domain` (pass
//! AND finished eval, K in {2, 5, 8}, nf in {1, 3}, mixed depths, β-carrying
//! blocks, a `beta_pc` domain falling back) and
//! `expand_batch_trajectory_hash_unchanged` (a whole tree, stacking on / off /
//! independent, identical trajectory hash).
//!
//! # When a domain does NOT join the stack (and runs its own pass, exactly as
//! before)
//!
//! * its gates carry per-column β (`beta_pc`): those columns index the domain's
//!   own seed and the stacked arm refuses them — `fallback_beta_pc`;
//! * the certified GPU seam could engage on the main pass (the lane armed, or
//!   the eval site shadowed): the per-domain call is what reaches the device —
//!   `fallback_seam`;
//! * it is alone in its chunk: the stackable domains are cut, in batch order,
//!   into chunks of at most `MAX_STACK_COLS` = 256 seed columns (one stacked
//!   pass per chunk of two or more, chunks concurrent) — `solo`. The cap never
//!   binds on cifar100 (K = 20 domains of nf <= 3); it bounds the single
//!   allocation on wide-nf tinyimagenet rows;
//! * the stacked pass returned `Err` (non-finite rows anywhere in the stack, a
//!   layout refusal): every block re-runs alone, which reproduces each
//!   domain's own result, error or not — `fallback_err`;
//! * the trusted constant `DOMAIN_STACK_MAIN_EVAL` is off — `disabled`.
//!
//! # Not stacked: the β / domain-alpha TRIAL passes
//!
//! Feasible — within one iteration index the domains' trial seeds and gates
//! are independent, and a shared-β / domain-alpha trial pass is column-local
//! by the same argument — but each ascent is a sequential accept/reject chain
//! (Polyak λ memory, direction refresh from the accepted pass, deadline poll
//! per iteration), so stacking them means driving both ascents as lock-step,
//! resumable per-domain state machines across the batch. The per-column β
//! trials also carry `beta_pc` and would need a block-local column shift in
//! the stacked arm. Left per-domain until the census says trials matter
//! (`site.beta_shared` / `site.dalpha` against `site.eval_stk`).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

use ndarray::{s, Array2};
use ny_core::{NyError, Result};
use rayon::prelude::*;

use super::{
    ChildProto, DomainEntry, EvalOut, EvalPrep, EvalSeeded, ExpandStep, MarginRowBab, YPack,
};
use crate::margin_row::engine::{Exceptions, LaneDir, PassOut, RowDomainGateBlock, Seed};

// --------------------------------------------------------------------------
//  The batch's staged evaluation
// --------------------------------------------------------------------------

/// Widest stacked main pass, in seed columns. The certified pass is already
/// linear in columns well below this (the B1 probe's cost curve flattens by
/// R ~ 64), so a wider stack buys no CPU and only grows the largest single
/// allocation — which matters on tinyimagenet rows, whose eval width is the
/// root open-class count (up to 99): K = 20 such domains would otherwise be a
/// ~2000-column pass. The stackable domains are cut, in batch order, into
/// consecutive chunks of at most this many columns (a domain wider than the cap
/// runs alone). A chunk is itself a valid stack, so the cut cannot move a bit.
pub(crate) const MAX_STACK_COLS: usize = 256;

/// Cut `widths` (in order) into consecutive index ranges whose widths sum to at
/// most `cap`; an item wider than `cap` gets a range of its own.
pub(crate) fn stack_chunks(widths: &[usize], cap: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut sum = 0usize;
    for (i, &w) in widths.iter().enumerate() {
        if i > start && sum.saturating_add(w) > cap {
            out.push(start..i);
            start = i;
            sum = 0;
        }
        sum = sum.saturating_add(w);
    }
    if start < widths.len() {
        out.push(start..widths.len());
    }
    out
}

/// Is the certified GPU seam of the eval main pass fully idle in this process?
/// Only then does a domain's own `run_collect_seamed` call reduce to the plain
/// CPU `run_collect` a stacked block reproduces; otherwise the per-domain call
/// is what reaches the device (or its shadow), so the domain keeps it.
fn eval_seam_idle() -> bool {
    use crate::margin_row::gpu_backward as gb;
    !gb::enabled() && gb::site_mode(gb::SITE_EVAL) == gb::SiteMode::Off
}

impl MarginRowBab<'_> {
    /// Evaluate a whole frontier batch with ONE stacked main pass (see the
    /// module docs). Slot `i` is exactly the `Result` that
    /// `self.eval_with_pack(&packs[i], &batch[i])` returns.
    pub(super) fn eval_batch_stacked(
        &self,
        batch: &[DomainEntry],
        packs: &[Arc<YPack>],
    ) -> (Vec<Result<EvalOut>>, StackReport) {
        let mut report = StackReport {
            batches: 1,
            ..StackReport::default()
        };
        if !self.cfg.domain_stack_main_eval {
            // Kill switch: the established per-domain evaluation, verbatim.
            report.disabled = batch.len() as u64;
            let evals: Vec<Result<EvalOut>> = batch
                .par_iter()
                .zip(packs.par_iter())
                .map(|(entry, pack)| self.eval_with_pack(pack, entry))
                .collect();
            return (evals, report);
        }
        // Stage 1: everything before the main pass, per domain.
        let preps: Vec<Result<EvalPrep>> = batch
            .par_iter()
            .zip(packs.par_iter())
            .map(|(entry, pack)| self.eval_prepare(pack, entry))
            .collect();
        let mut outs: Vec<Option<Result<EvalOut>>> = Vec::with_capacity(batch.len());
        let mut seeded: Vec<(usize, Box<EvalSeeded>)> = Vec::new();
        for (i, prep) in preps.into_iter().enumerate() {
            match prep {
                Ok(EvalPrep::Done(out)) => outs.push(Some(Ok(*out))),
                Ok(EvalPrep::Seeded(one)) => {
                    outs.push(None);
                    seeded.push((i, one));
                }
                Err(e) => outs.push(Some(Err(e))),
            }
        }
        // Stage 2: the main passes — one stacked pass where admissible.
        let (passes, stage) = {
            let refs: Vec<&EvalSeeded> = seeded.iter().map(|(_, one)| one.as_ref()).collect();
            self.eval_main_passes(&refs)
        };
        report.absorb(&stage);
        // Stage 3: everything after the main pass, per domain.
        let finished: Vec<(usize, Result<EvalOut>)> = seeded
            .into_par_iter()
            .zip(passes.into_par_iter())
            .map(|((i, one), pass)| {
                let out = pass.and_then(|pass| self.eval_finish(&batch[i], one, pass));
                (i, out)
            })
            .collect();
        for (i, out) in finished {
            outs[i] = Some(out);
        }
        let evals: Vec<Result<EvalOut>> = outs
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    Err(NyError::InvalidSpec(
                        "margin_row: stacked eval slot left unfilled".into(),
                    ))
                })
            })
            .collect();
        (evals, report)
    }

    /// The main certified pass of every `seeded` domain, aligned with the
    /// input: ONE stacked pass over the stackable domains, each other domain's
    /// own pass ([`MarginRowBab::eval_main_pass`], the unsplit body's call).
    pub(super) fn eval_main_passes(
        &self,
        seeded: &[&EvalSeeded],
    ) -> (Vec<Result<PassOut>>, StackReport) {
        let n = seeded.len();
        let mut report = StackReport::default();
        let mut out: Vec<Option<Result<PassOut>>> = (0..n).map(|_| None).collect();
        let mut stackable: Vec<usize> = Vec::with_capacity(n);
        if eval_seam_idle() {
            for (i, one) in seeded.iter().enumerate() {
                if one.dom.beta_pc.is_empty() {
                    stackable.push(i);
                } else {
                    report.fallback_beta_pc += 1;
                }
            }
        } else {
            report.fallback_seam = n as u64;
        }
        // Consecutive chunks of at most `MAX_STACK_COLS` columns; each chunk of
        // two or more domains is one stacked pass (chunks run concurrently),
        // a chunk of one runs its own pass.
        let widths: Vec<usize> = stackable
            .iter()
            .map(|&i| seeded[i].main_seed().s.ncols())
            .collect();
        let chunks: Vec<Vec<usize>> = stack_chunks(&widths, MAX_STACK_COLS)
            .into_iter()
            .map(|range| stackable[range].to_vec())
            .collect();
        let stacked: Vec<_> = chunks
            .par_iter()
            .filter(|chunk| chunk.len() >= 2)
            .map(|chunk| (chunk, self.stacked_main_pass(seeded, chunk)))
            .collect();
        report.solo = chunks.iter().filter(|chunk| chunk.len() < 2).count() as u64;
        for (chunk, result) in stacked {
            match result {
                Ok((passes, total)) => {
                    let k = chunk.len() as u64;
                    report.passes += 1;
                    report.blocks += k;
                    report.k_max = report.k_max.max(k);
                    report.cols += total as u64;
                    report.cols_max = report.cols_max.max(total as u64);
                    for (&i, pass) in chunk.iter().zip(passes) {
                        out[i] = Some(Ok(pass));
                    }
                }
                Err(_) => report.fallback_err += chunk.len() as u64,
            }
        }
        // Everything not served by the stack runs its own pass, concurrently
        // (the per-domain lane's own concurrency).
        let rest: Vec<usize> = (0..n).filter(|&i| out[i].is_none()).collect();
        let own: Vec<(usize, Result<PassOut>)> = rest
            .par_iter()
            .map(|&i| (i, self.eval_main_pass(seeded[i])))
            .collect();
        for (i, pass) in own {
            out[i] = Some(pass);
        }
        let passes: Vec<Result<PassOut>> = out
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    Err(NyError::InvalidSpec(
                        "margin_row: stacked main pass slot left unfilled".into(),
                    ))
                })
            })
            .collect();
        (passes, report)
    }

    /// ONE stacked main pass over `seeded[idx]` (in `idx` order), sliced back
    /// into per-domain passes; also returns the stacked width. Any `Err` —
    /// including a seed-shape disagreement between the domains — sends every
    /// block back to its own pass.
    fn stacked_main_pass(
        &self,
        seeded: &[&EvalSeeded],
        idx: &[usize],
    ) -> Result<(Vec<PassOut>, usize)> {
        let refuse = |why: &str| NyError::InvalidSpec(format!("margin_row: mr-stack {why}"));
        let first = seeded[idx[0]].main_seed();
        let n_y = first.s.nrows();
        let with_e = first.e.is_some();
        let mut total = 0usize;
        for &i in idx {
            let one = seeded[i].main_seed();
            if one.s.nrows() != n_y || one.e.is_some() != with_e || one.s.ncols() == 0 {
                return Err(refuse("seed shapes disagree"));
            }
            total = total
                .checked_add(one.s.ncols())
                .ok_or_else(|| refuse("width overflow"))?;
        }
        // Concatenate the seeds column-wise, bit-for-bit.
        let mut stacked_s = Array2::<f64>::zeros((n_y, total));
        let mut stacked_e = with_e.then(|| Array2::<f64>::zeros((n_y, total)));
        let mut offset = 0usize;
        for &i in idx {
            let one = seeded[i].main_seed();
            let w = one.s.ncols();
            stacked_s
                .slice_mut(s![.., offset..offset + w])
                .assign(&one.s);
            if let (Some(dst), Some(src)) = (stacked_e.as_mut(), one.e.as_ref()) {
                if src.raw_dim() != one.s.raw_dim() {
                    return Err(refuse("seed error-lane shape mismatch"));
                }
                dst.slice_mut(s![.., offset..offset + w]).assign(src);
            }
            offset += w;
        }
        // Each domain owns its contiguous column range under its own gates
        // (inherited β installed; no candidate exceptions on a main pass).
        let no_exc = Exceptions::default();
        let mut start = 0usize;
        let blocks: Vec<RowDomainGateBlock<'_>> = idx
            .iter()
            .map(|&i| {
                let w = seeded[i].main_seed().s.ncols();
                let block = RowDomainGateBlock {
                    columns: start..start + w,
                    gates: &seeded[i].dom,
                    exceptions: &no_exc,
                };
                start += w;
                block
            })
            .collect();
        // One collection spec for the stack: `unst_rows` is captured when ANY
        // block's eval wants it and dropped below for the blocks that did not
        // ask (the capture is a read-only pre-transform copy; it never moves
        // a coefficient).
        let want_unst_rows: Vec<bool> = idx.iter().map(|&i| seeded[i].unst_rows()).collect();
        let collect = self.eval_collect(want_unst_rows.iter().any(|&w| w));
        let mut passes = {
            let _t_pass =
                crate::margin_row::prof::Timer::start(crate::margin_row::prof::Phase::EvalPass);
            let _census = crate::margin_row::census::SiteTimer::start(
                crate::margin_row::census::Site::EvalStacked,
                total,
            );
            self.eng.run_domain_stacked_split(
                &Seed {
                    s: stacked_s,
                    e: stacked_e,
                },
                &blocks,
                LaneDir::Lower,
                collect,
            )?
        };
        if passes.len() != idx.len() {
            return Err(refuse("split count mismatch"));
        }
        for (pass, &want) in passes.iter_mut().zip(&want_unst_rows) {
            if !want {
                pass.unst_rows = None;
            }
        }
        Ok((passes, total))
    }
}

// --------------------------------------------------------------------------
//  Receipt: per batch, per tree, per process
// --------------------------------------------------------------------------

/// One batch's (or a tree's accumulated) main-eval stacking counts. Telemetry
/// only; nothing reads it back into a decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StackReport {
    /// Frontier batches that took the domain-stacked evaluation route.
    pub(crate) batches: u64,
    /// Stacked main passes that returned (each serves >= 2 domains).
    pub(crate) passes: u64,
    /// Domains served by stacked passes (sum of K over passes).
    pub(crate) blocks: u64,
    /// Widest stack (max K).
    pub(crate) k_max: u64,
    /// Seed columns across stacked passes (sum of R over passes).
    pub(crate) cols: u64,
    /// Widest stacked pass (max R).
    pub(crate) cols_max: u64,
    /// Domains that ran their own pass because their gates carry `beta_pc`.
    pub(crate) fallback_beta_pc: u64,
    /// Domains that ran their own pass because the GPU seam could engage.
    pub(crate) fallback_seam: u64,
    /// Domains re-run alone after a stacked pass returned `Err`.
    pub(crate) fallback_err: u64,
    /// Domains that ran their own pass alone in their column-capped chunk.
    pub(crate) solo: u64,
    /// Domains evaluated per domain because `DOMAIN_STACK_MAIN_EVAL` is off.
    pub(crate) disabled: u64,
}

impl StackReport {
    /// Accumulate `o` (sums; maxima for the two widths).
    pub(crate) fn absorb(&mut self, o: &Self) {
        self.batches += o.batches;
        self.passes += o.passes;
        self.blocks += o.blocks;
        self.k_max = self.k_max.max(o.k_max);
        self.cols += o.cols;
        self.cols_max = self.cols_max.max(o.cols_max);
        self.fallback_beta_pc += o.fallback_beta_pc;
        self.fallback_seam += o.fallback_seam;
        self.fallback_err += o.fallback_err;
        self.solo += o.solo;
        self.disabled += o.disabled;
    }
}

static STK_BATCHES: AtomicU64 = AtomicU64::new(0);
static STK_PASSES: AtomicU64 = AtomicU64::new(0);
static STK_BLOCKS: AtomicU64 = AtomicU64::new(0);
static STK_K_MAX: AtomicU64 = AtomicU64::new(0);
static STK_COLS: AtomicU64 = AtomicU64::new(0);
static STK_COLS_MAX: AtomicU64 = AtomicU64::new(0);
static STK_FB_BETA_PC: AtomicU64 = AtomicU64::new(0);
static STK_FB_SEAM: AtomicU64 = AtomicU64::new(0);
static STK_FB_ERR: AtomicU64 = AtomicU64::new(0);
static STK_SOLO: AtomicU64 = AtomicU64::new(0);
static STK_DISABLED: AtomicU64 = AtomicU64::new(0);
/// The OUTERMOST parallel tree's trajectory hash / merged decisions, and how
/// many trees published one (the last publication wins).
static TRAJ_HASH: AtomicU64 = AtomicU64::new(0);
static TRAJ_STEPS: AtomicU64 = AtomicU64::new(0);
static TRAJ_TREES: AtomicU64 = AtomicU64::new(0);

fn note_global(r: &StackReport) {
    STK_BATCHES.fetch_add(r.batches, Relaxed);
    STK_PASSES.fetch_add(r.passes, Relaxed);
    STK_BLOCKS.fetch_add(r.blocks, Relaxed);
    STK_K_MAX.fetch_max(r.k_max, Relaxed);
    STK_COLS.fetch_add(r.cols, Relaxed);
    STK_COLS_MAX.fetch_max(r.cols_max, Relaxed);
    STK_FB_BETA_PC.fetch_add(r.fallback_beta_pc, Relaxed);
    STK_FB_SEAM.fetch_add(r.fallback_seam, Relaxed);
    STK_FB_ERR.fetch_add(r.fallback_err, Relaxed);
    STK_SOLO.fetch_add(r.solo, Relaxed);
    STK_DISABLED.fetch_add(r.disabled, Relaxed);
}

fn global_report() -> StackReport {
    StackReport {
        batches: STK_BATCHES.load(Relaxed),
        passes: STK_PASSES.load(Relaxed),
        blocks: STK_BLOCKS.load(Relaxed),
        k_max: STK_K_MAX.load(Relaxed),
        cols: STK_COLS.load(Relaxed),
        cols_max: STK_COLS_MAX.load(Relaxed),
        fallback_beta_pc: STK_FB_BETA_PC.load(Relaxed),
        fallback_seam: STK_FB_SEAM.load(Relaxed),
        fallback_err: STK_FB_ERR.load(Relaxed),
        solo: STK_SOLO.load(Relaxed),
        disabled: STK_DISABLED.load(Relaxed),
    }
}

/// Reset the process-global `[mr-stack]` counters (lane start; `census::reset`).
pub(crate) fn mr_stack_reset() {
    for a in [
        &STK_BATCHES,
        &STK_PASSES,
        &STK_BLOCKS,
        &STK_K_MAX,
        &STK_COLS,
        &STK_COLS_MAX,
        &STK_FB_BETA_PC,
        &STK_FB_SEAM,
        &STK_FB_ERR,
        &STK_SOLO,
        &STK_DISABLED,
        &TRAJ_HASH,
        &TRAJ_STEPS,
        &TRAJ_TREES,
    ] {
        a.store(0, Relaxed);
    }
}

/// The `[mr-stack]` stop receipt, printed after `[mr-census]` at EVERY lane
/// stop, zero or not — so `passes=0` (stacking never engaged: `domain_stack`
/// dark, width-1 batches, or every domain fell back — the `fallback_*` / `solo`
/// fields say which) is distinguishable from a binary without B5 (no line).
///
/// `K` = widest stack (domains in one pass); `nf` = seed columns summed over
/// stacked passes; `blocks` = domains served by stacked passes. So
/// `blocks / passes` is the mean stack depth and `nf / blocks` the mean eval
/// width per stacked domain. `traj` / `traj_steps` are the outermost parallel
/// tree's trajectory hash and merged-decision count (`traj_trees=0`: no
/// parallel tree published one).
pub(crate) fn mr_stack_stop_line() -> String {
    let r = global_report();
    format!(
        "[mr-stack] main_eval K={} nf={} blocks={} passes={} batches={} cols_max={} \
         fallback_beta_pc={} fallback_seam={} fallback_err={} solo={} disabled={} engaged={} \
         traj={:#018x} traj_steps={} traj_trees={}",
        r.k_max,
        r.cols,
        r.blocks,
        r.passes,
        r.batches,
        r.cols_max,
        r.fallback_beta_pc,
        r.fallback_seam,
        r.fallback_err,
        r.solo,
        r.disabled,
        u8::from(r.passes > 0),
        TRAJ_HASH.load(Relaxed),
        TRAJ_STEPS.load(Relaxed),
        TRAJ_TREES.load(Relaxed),
    )
}

/// The same counts as `[mr-census]` fields. These are the observable on a
/// row whose lane is still running at process exit (the detached concurrent
/// lane on every timeout row): that path prints only the SNAPSHOT
/// `[mr-census]` line, never the `[mr-stack]` stop line.
pub(crate) fn mr_stack_fields() -> String {
    let r = global_report();
    format!(
        " stk_batches={} stk_passes={} stk_blocks={} stk_cols={} stk_k_max={} \
         stk_fallback_beta_pc={} stk_fallback_seam={} stk_fallback_err={} stk_solo={} \
         stk_disabled={}",
        r.batches,
        r.passes,
        r.blocks,
        r.cols,
        r.k_max,
        r.fallback_beta_pc,
        r.fallback_seam,
        r.fallback_err,
        r.solo,
        r.disabled,
    )
}

// --------------------------------------------------------------------------
//  Per-tree trace: the receipt plus the trajectory hash
// --------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Trajectory tag: a popped entry whose queued bound already closed.
pub(super) const TRAJ_QUEUED: u64 = 0x51;
/// Trajectory tag: a Tier-2 epoch subtree closed the popped entry.
pub(super) const TRAJ_EPOCH: u64 = 0x52;
const TRAJ_INFEASIBLE: u64 = 0x61;
const TRAJ_DEADLINE: u64 = 0x62;
const TRAJ_CLOSED_AT_EVAL: u64 = 0x63;
const TRAJ_NO_CANDIDATES: u64 = 0x64;
const TRAJ_NON_FINITE: u64 = 0x65;
const TRAJ_EXPANDED: u64 = 0x66;

/// One tree's main-eval stacking receipt and trajectory hash.
///
/// The TRAJECTORY HASH is FNV-1a over every decision the parallel tree loop
/// merges, in merge order: the popped domain (bound bits, trunk and head
/// splits), the step kind, and for an expansion its dips, closed / created
/// counts, close kinds and every pushed child's bound, splits, y-box, β state,
/// retirement set and carried alpha, all as exact bits. Two runs with equal
/// hashes merged the same tree bit-for-bit. Telemetry only: a few integer ops
/// per merged decision, never read by any decision. (Time-driven policies —
/// deadlines, the closure ladder's clock — legitimately diverge between runs,
/// so compare hashes only between deterministic runs.)
#[derive(Debug, Clone)]
pub(crate) struct B5Trace {
    traj: u64,
    traj_steps: u64,
    report: StackReport,
}

impl Default for B5Trace {
    fn default() -> Self {
        Self {
            traj: FNV_OFFSET,
            traj_steps: 0,
            report: StackReport::default(),
        }
    }
}

impl B5Trace {
    fn word(&mut self, w: u64) {
        for byte in w.to_le_bytes() {
            self.traj ^= u64::from(byte);
            self.traj = self.traj.wrapping_mul(FNV_PRIME);
        }
    }

    fn f64s(&mut self, v: &[f64]) {
        self.word(v.len() as u64);
        for x in v {
            self.word(x.to_bits());
        }
    }

    fn splits(&mut self, trunk: &[super::TrunkSplit], heads: &[super::HeadFix]) {
        self.word(trunk.len() as u64);
        for &(li, pos, dir) in trunk {
            self.word(li as u64);
            self.word(pos as u64);
            self.word(dir as u64);
        }
        self.word(heads.len() as u64);
        for &(i, dir) in heads {
            self.word(i as u64);
            self.word(dir as u64);
        }
    }

    fn entry(&mut self, e: &DomainEntry) {
        self.word(e.bound.to_bits());
        self.splits(&e.trunk, &e.heads);
    }

    fn child(&mut self, c: &ChildProto) {
        self.word(c.bound.to_bits());
        self.splits(&c.trunk, &c.heads);
        self.f64s(&c.ly);
        self.f64s(&c.uy);
        self.f64s(&c.betas);
        self.f64s(&c.head_betas);
        self.word(c.beta_step.to_bits());
        self.word(c.pc.len() as u64);
        for s in &c.pc {
            self.word(s.col as u64);
            self.f64s(&s.trunk);
            self.f64s(&s.heads);
        }
        self.word(c.retired.len() as u64);
        for &r in &c.retired {
            self.word(u64::from(r));
        }
        self.word(c.alpha.len() as u64);
        for &(li, idx, a) in &c.alpha {
            self.word(u64::from(li));
            self.word(u64::from(idx));
            self.word(a.to_bits());
        }
        self.word(u64::from(c.clip_acc.is_some()));
    }

    /// Fold a popped entry closed in the fill stage (`TRAJ_QUEUED` /
    /// `TRAJ_EPOCH`).
    pub(super) fn fold_close(&mut self, tag: u64, entry: &DomainEntry) {
        self.traj_steps += 1;
        self.word(tag);
        self.entry(entry);
    }

    /// Fold one merged expansion step and the entry it came from.
    pub(super) fn fold_step(&mut self, step: &ExpandStep, entry: &DomainEntry) {
        self.traj_steps += 1;
        self.entry(entry);
        match step {
            ExpandStep::Infeasible(_, _) => self.word(TRAJ_INFEASIBLE),
            ExpandStep::DeadlineHit(_) => self.word(TRAJ_DEADLINE),
            ExpandStep::ClosedAtEval(_, _, _) => self.word(TRAJ_CLOSED_AT_EVAL),
            ExpandStep::NoCandidates(_) => self.word(TRAJ_NO_CANDIDATES),
            ExpandStep::NonFiniteBound(_) => self.word(TRAJ_NON_FINITE),
            ExpandStep::Expanded {
                dips,
                worst_dip,
                pushes,
                closed_children,
                created_children,
                close_kinds,
                closed_leaves: _,
                execution: _,
                hints: _,
            } => {
                self.word(TRAJ_EXPANDED);
                self.word(*dips as u64);
                self.word(worst_dip.to_bits());
                self.word(*closed_children as u64);
                self.word(*created_children as u64);
                self.word(close_kinds.raw as u64);
                self.word(close_kinds.lift as u64);
                self.word(close_kinds.head_infeasible as u64);
                self.word(pushes.len() as u64);
                for c in pushes {
                    self.child(c);
                }
            }
        }
    }

    /// Absorb one batch's stacking receipt (per tree and process-wide).
    pub(super) fn note_batch(&mut self, r: StackReport) {
        self.report.absorb(&r);
        note_global(&r);
    }

    /// Publish this (outermost) tree's hash and receipt at its stop.
    pub(super) fn publish(&self) {
        TRAJ_HASH.store(self.traj, Relaxed);
        TRAJ_STEPS.store(self.traj_steps, Relaxed);
        TRAJ_TREES.fetch_add(1, Relaxed);
        #[cfg(test)]
        LAST.with(|cell| cell.set(Some((self.traj, self.traj_steps, self.report))));
    }
}

#[cfg(test)]
thread_local! {
    /// The last OUTERMOST parallel tree on this thread: `(hash, steps,
    /// receipt)`. Tests run concurrently and the process-global counters mix
    /// them; the tree loop runs on the calling test thread, so this does not.
    static LAST: std::cell::Cell<Option<(u64, u64, StackReport)>> =
        const { std::cell::Cell::new(None) };
}

/// TEST-ONLY: take the last outermost parallel tree's `(trajectory hash,
/// merged decisions, stacking receipt)` published on THIS thread.
#[cfg(test)]
pub(crate) fn take_last_trajectory() -> Option<(u64, u64, StackReport)> {
    LAST.with(std::cell::Cell::take)
}

// --------------------------------------------------------------------------
//  Test-only differential: stacked main pass vs each domain's own
// --------------------------------------------------------------------------

/// What [`MarginRowBab::stacked_main_eval_differential`] observed.
#[cfg(test)]
pub(crate) struct StackedEvalDiff {
    /// Split depth of every evaluated domain, in stack order.
    pub(crate) depths: Vec<usize>,
    /// Main-pass width (seed columns) of every evaluated domain.
    pub(crate) widths: Vec<usize>,
    /// Index of the domain whose gates carried injected `beta_pc`, if any.
    pub(crate) beta_pc_index: Option<usize>,
    /// Domains whose gates carried injected shared β.
    pub(crate) beta_blocks: usize,
    /// Stack-eligible domains (no `beta_pc`) whose eval requested the
    /// `unst_rows` gradient capture; the other eligible domains get it DROPPED
    /// from their slice of the stacked pass.
    pub(crate) capture_blocks: usize,
    /// The stage's receipt.
    pub(crate) report: StackReport,
    /// Every bit-level disagreement found (empty = identical).
    pub(crate) mismatches: Vec<String>,
    /// Domains compared (pass AND finished eval).
    pub(crate) compared: usize,
}

#[cfg(test)]
fn entry_from_child(c: ChildProto) -> DomainEntry {
    DomainEntry {
        bound: c.bound,
        seq: 0,
        trunk: c.trunk,
        heads: c.heads,
        ly: c.ly,
        uy: c.uy,
        betas: c.betas,
        head_betas: c.head_betas,
        beta_step: c.beta_step,
        pc: c.pc,
        retired: c.retired,
        alpha: c.alpha,
        clip_acc: c.clip_acc,
        split_order: c.split_order,
    }
}

#[cfg(test)]
fn bits_eq(tag: &str, what: &str, a: &[f64], b: &[f64], out: &mut Vec<String>) {
    if a.len() != b.len() {
        out.push(format!("{tag}: {what} length {} vs {}", a.len(), b.len()));
        return;
    }
    if let Some(i) = (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits()) {
        out.push(format!("{tag}: {what}[{i}] {:e} vs {:e}", a[i], b[i]));
    }
}

#[cfg(test)]
fn mat_eq(tag: &str, what: &str, a: &Array2<f64>, b: &Array2<f64>, out: &mut Vec<String>) {
    if a.dim() != b.dim() {
        out.push(format!("{tag}: {what} dim {:?} vs {:?}", a.dim(), b.dim()));
        return;
    }
    if let Some(((i, j), x)) = a
        .indexed_iter()
        .find(|((i, j), x)| x.to_bits() != b[[*i, *j]].to_bits())
    {
        out.push(format!("{tag}: {what}[{i},{j}] {x:e} vs {:e}", b[[i, j]]));
    }
}

#[cfg(test)]
fn mat_map_eq(
    tag: &str,
    what: &str,
    a: Option<&std::collections::BTreeMap<usize, Array2<f64>>>,
    b: Option<&std::collections::BTreeMap<usize, Array2<f64>>>,
    out: &mut Vec<String>,
) {
    match (a, b) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            if a.keys().ne(b.keys()) {
                out.push(format!("{tag}: {what} layer keys differ"));
                return;
            }
            for (li, ma) in a {
                mat_eq(tag, &format!("{what}[{li}]"), ma, &b[li], out);
            }
        }
        _ => out.push(format!("{tag}: {what} presence differs")),
    }
}

#[cfg(test)]
fn vec_map_eq(
    tag: &str,
    what: &str,
    a: &std::collections::BTreeMap<usize, Vec<f64>>,
    b: &std::collections::BTreeMap<usize, Vec<f64>>,
    out: &mut Vec<String>,
) {
    if a.keys().ne(b.keys()) {
        out.push(format!("{tag}: {what} layer keys differ"));
        return;
    }
    for (li, va) in a {
        bits_eq(tag, &format!("{what}[{li}]"), va, &b[li], out);
    }
}

/// Bit-compare two passes field by field (coefficients, error lane, bias,
/// bias error, `unst_abs` mass, Tier-0 and `unst_rows` captures).
#[cfg(test)]
fn pass_eq(tag: &str, a: &PassOut, b: &PassOut, out: &mut Vec<String>) {
    mat_eq(tag, "a", &a.a, &b.a, out);
    match (&a.e, &b.e) {
        (None, None) => {}
        (Some(ea), Some(eb)) => mat_eq(tag, "e", ea, eb, out),
        _ => out.push(format!("{tag}: error-lane presence differs")),
    }
    bits_eq(tag, "b", &a.b, &b.b, out);
    bits_eq(tag, "eb", &a.eb, &b.eb, out);
    match (&a.coll, &b.coll) {
        (None, None) => {}
        (Some(ca), Some(cb)) => vec_map_eq(tag, "coll", ca, cb, out),
        _ => out.push(format!("{tag}: coll presence differs")),
    }
    mat_map_eq(
        tag,
        "coll_rows",
        a.coll_rows.as_ref(),
        b.coll_rows.as_ref(),
        out,
    );
    mat_map_eq(
        tag,
        "unst_rows",
        a.unst_rows.as_ref(),
        b.unst_rows.as_ref(),
        out,
    );
}

/// Bit-compare two finished evaluations: the bound and every piece of node
/// state a child, the scorer or the serial merge reads.
#[cfg(test)]
fn eval_eq(tag: &str, a: &EvalOut, b: &EvalOut, out: &mut Vec<String>) {
    match (a, b) {
        (EvalOut::Node(ba, sa), EvalOut::Node(bb, sb)) => {
            if ba.to_bits() != bb.to_bits() {
                out.push(format!("{tag}: bound {ba:e} vs {bb:e}"));
            }
            bits_eq(tag, "ybox.ly", &sa.ybox.ly, &sb.ybox.ly, out);
            bits_eq(tag, "ybox.uy", &sa.ybox.uy, &sb.ybox.uy, out);
            bits_eq(tag, "betas", &sa.betas, &sb.betas, out);
            bits_eq(tag, "head_betas", &sa.head_betas, &sb.head_betas, out);
            if sa.beta_step.to_bits() != sb.beta_step.to_bits() {
                out.push(format!("{tag}: beta_step differs"));
            }
            if sa.retired != sb.retired {
                out.push(format!(
                    "{tag}: retired {:?} vs {:?}",
                    sa.retired, sb.retired
                ));
            }
            let alpha_bits = |v: &[(u32, u32, f64)]| -> Vec<(u32, u32, u64)> {
                v.iter().map(|&(l, i, x)| (l, i, x.to_bits())).collect()
            };
            if alpha_bits(sa.alpha.as_slice()) != alpha_bits(sb.alpha.as_slice()) {
                out.push(format!("{tag}: carried alpha differs"));
            }
            if sa.pc.len() != sb.pc.len() {
                out.push(format!("{tag}: pc length differs"));
            } else {
                for (pa, pb) in sa.pc.iter().zip(&sb.pc) {
                    if pa.col != pb.col {
                        out.push(format!("{tag}: pc column differs"));
                    }
                    bits_eq(tag, "pc.trunk", &pa.trunk, &pb.trunk, out);
                    bits_eq(tag, "pc.heads", &pa.heads, &pb.heads, out);
                }
            }
            if sa.policy_execution != sb.policy_execution {
                out.push(format!("{tag}: policy execution evidence differs"));
            }
            vec_map_eq(tag, "st.coll", &sa.coll, &sb.coll, out);
            mat_map_eq(
                tag,
                "st.coll_rows",
                Some(&sa.coll_rows),
                Some(&sb.coll_rows),
                out,
            );
            pass_eq(&format!("{tag} st.pass"), &sa.pass, &sb.pass, out);
            if sa.dom.layers.keys().ne(sb.dom.layers.keys()) {
                out.push(format!("{tag}: dom layer keys differ"));
            } else {
                for (li, ga) in &sa.dom.layers {
                    let gb = &sb.dom.layers[li];
                    bits_eq(tag, "dom.alpha", &ga.alpha, &gb.alpha, out);
                    bits_eq(tag, "dom.s", &ga.s, &gb.s, out);
                    bits_eq(tag, "dom.c", &ga.c, &gb.c, out);
                    bits_eq(tag, "dom.ms", &ga.ms, &gb.ms, out);
                }
            }
            let beta_bits =
                |d: &crate::margin_row::engine::DomainGates| -> Vec<(usize, usize, i8, u64)> {
                    d.beta
                        .iter()
                        .flat_map(|(&li, ts)| {
                            ts.iter()
                                .map(move |t| (li, t.neuron, t.sign, t.beta.to_bits()))
                        })
                        .collect()
                };
            if beta_bits(&sa.dom) != beta_bits(&sb.dom) {
                out.push(format!("{tag}: dom beta terms differ"));
            }
        }
        (EvalOut::Infeasible(_, _), EvalOut::Infeasible(_, _))
        | (EvalOut::Deadline(_), EvalOut::Deadline(_)) => {}
        _ => out.push(format!("{tag}: EvalOut variants differ")),
    }
}

#[cfg(test)]
impl<'a> MarginRowBab<'a> {
    /// A bare driver over `(net, root, t, adv)` with `cfg`, plus its root heap
    /// entry (test fixture; mirrors the production driver construction).
    fn stack_test_driver(
        net: &'a crate::margin_row::TwinNet,
        root: &'a crate::margin_row::RootGates,
        t: usize,
        adv: &[usize],
        cfg: super::BabConfig,
    ) -> (Self, DomainEntry) {
        use super::{
            BabStats, EpochExecutionEvidence, FrontierExecutionEvidence, Ledger, Lru,
            PolicyExecutionEvidence, WidthExecutionEvidence,
        };
        let eng = crate::margin_row::engine::BackwardEngine::new(net, root);
        let re = super::root_eval(&eng, net, t, adv).expect("root_eval");
        let mb = crate::margin_row::bounds::MarginBatch::new(net, t, &re.tree_classes)
            .expect("margin batch");
        let root_bound = re
            .dj
            .iter()
            .zip(adv)
            .filter(|(_, j)| re.tree_classes.contains(j))
            .map(|(b, _)| *b)
            .fold(f64::INFINITY, f64::min);
        let lru_cap = cfg.lru_cap.max(1);
        let stats = BabStats {
            root_bound,
            tree_classes: re.tree_classes.clone(),
            root_closed_classes: 0,
            expansions: 0,
            domains_created: 1,
            closed: 0,
            max_depth: 0,
            mono_raw_dips: 0,
            mono_worst: 0.0,
            stop: String::new(),
            elapsed_secs: 0.0,
            class_runs: Vec::new(),
            epochs_attempted: 0,
            epochs_closed: 0,
            ledger_ok: None,
            frontier_execution: FrontierExecutionEvidence::joint(cfg.frontier, cfg.frontier),
            policy_execution: PolicyExecutionEvidence::from_config(&cfg),
            epoch_execution: EpochExecutionEvidence::from_config(&cfg),
            width_execution: WidthExecutionEvidence::from_config(&cfg),
            root_precision_execution: cfg.root_precision_execution,
            backward_interm_execution: cfg.backward_interm_execution,
            root_alpha_execution: cfg.root_alpha_execution,
            close_integrity: super::CloseIntegrityEvidence::default(),
        };
        let mut bab = Self {
            eng,
            mb,
            cfg,
            lru: Lru {
                cap: lru_cap,
                entries: Vec::new(),
            },
            stats,
            epoch_failures: 0,
            ledger: Ledger::default(),
            nested_ledgers_ok: true,
            lane_cache: crate::margin_row::gpu_backward::cache::PlanCache::default(),
            closure: crate::margin_row::closure::ClosureMonitor::new(false),
            b5: B5Trace::default(),
        };
        bab.lru.put(Vec::new(), re.pack.clone());
        let entry = bab.root_entry(root_bound, &re, 0);
        (bab, entry)
    }

    /// The seeded half of one domain's eval, with the fixture's injected
    /// gates: shared β terms when `beta` (so β-carrying and β-free blocks mix
    /// in one stack), per-column β when `beta_pc`. The injected terms need not
    /// be a valid Lagrangian: this compares ARITHMETIC bit-for-bit between two
    /// dispatches of the same pass, not a bound's validity.
    ///
    /// Every third slot also has its β / domain-alpha ascents disarmed, so it
    /// does NOT request the `unst_rows` gradient capture. Without that, a
    /// stack's blocks all request it (domain-alpha on) or none do (off), and the
    /// stacked pass's per-block DROP (`pass.unst_rows = None` for a block that
    /// did not ask, exactly like its own pass) would never run in the
    /// differential. Both dispatches are prepared through this function, so the
    /// flip is identical on both sides.
    fn stack_test_seed(
        &self,
        pack: &Arc<YPack>,
        entry: &DomainEntry,
        slot: usize,
        beta: bool,
        beta_pc: bool,
    ) -> Option<Box<EvalSeeded>> {
        use crate::margin_row::engine::{BetaSplit, PcBetaCol};
        let EvalPrep::Seeded(mut one) = self.eval_prepare(pack, entry).expect("eval_prepare")
        else {
            return None;
        };
        let root = self.eng.root;
        let (li, neuron) = root
            .layers
            .iter()
            .enumerate()
            .find_map(|(li, rec)| rec.unst.first().map(|&idx| (li, idx)))?;
        if beta {
            let sign = if slot.is_multiple_of(2) { 1 } else { -1 };
            one.dom.beta.insert(
                li,
                vec![BetaSplit {
                    neuron,
                    sign,
                    beta: 0.02 * (slot + 1) as f64,
                }],
            );
        }
        if beta_pc {
            one.dom.beta_pc.insert(
                li,
                vec![PcBetaCol {
                    col: 0,
                    terms: vec![BetaSplit {
                        neuron,
                        sign: 1,
                        beta: 0.05,
                    }],
                }],
            );
        }
        if slot % 3 == 2 {
            one.beta_armed = false;
            one.alpha_armed = false;
        }
        Some(one)
    }

    /// DIFFERENTIAL (test-only): walk the tree breadth-first from the root to
    /// collect `k` domains at mixed depths (plus one more carrying `beta_pc`
    /// when `inject_beta_pc`), then evaluate every domain TWICE from
    /// independently prepared states — once through ONE stacked main pass
    /// ([`Self::eval_main_passes`]) and once through its own pass
    /// ([`Self::eval_main_pass`]) — and bit-compare both the passes and the
    /// finished evaluations (bound, node state, β / alpha state, captures).
    /// `inject_beta` installs shared β on every odd slot.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stacked_main_eval_differential(
        net: &'a crate::margin_row::TwinNet,
        root: &'a crate::margin_row::RootGates,
        t: usize,
        adv: &[usize],
        k: usize,
        cfg: super::BabConfig,
        inject_beta: bool,
        inject_beta_pc: bool,
    ) -> StackedEvalDiff {
        let (mut bab, root_entry) = Self::stack_test_driver(net, root, t, adv, cfg);
        let want = k + usize::from(inject_beta_pc);
        // Breadth-first: the root, its children, their children, ... so the
        // stack mixes depths (and, under column retirement, widths).
        let mut queue = std::collections::VecDeque::from([root_entry]);
        let mut entries: Vec<DomainEntry> = Vec::new();
        let mut packs: Vec<Arc<YPack>> = Vec::new();
        while entries.len() < want {
            let Some(e) = queue.pop_front() else { break };
            let pack = bab.rows_for(&e.trunk).expect("rows_for");
            if let Ok(ExpandStep::Expanded { pushes, .. }) = bab.expand_one(&pack, &e) {
                queue.extend(pushes.into_iter().map(entry_from_child));
            }
            // Only domains that reach their main pass can be stacked.
            if matches!(bab.eval_prepare(&pack, &e), Ok(EvalPrep::Seeded(_))) {
                entries.push(e);
                packs.push(pack);
            }
        }
        let beta_pc_index = want
            .checked_sub(1)
            .filter(|_| inject_beta_pc && entries.len() == want);
        let mut beta_blocks = 0usize;
        let mut stk: Vec<Box<EvalSeeded>> = Vec::new();
        let mut own: Vec<Box<EvalSeeded>> = Vec::new();
        for (slot, (entry, pack)) in entries.iter().zip(&packs).enumerate() {
            let beta = inject_beta && slot % 2 == 1;
            let pc = beta_pc_index == Some(slot);
            beta_blocks += usize::from(beta);
            stk.push(
                bab.stack_test_seed(pack, entry, slot, beta, pc)
                    .expect("fixture domain seeds"),
            );
            own.push(
                bab.stack_test_seed(pack, entry, slot, beta, pc)
                    .expect("fixture domain seeds"),
            );
        }
        let depths: Vec<usize> = entries
            .iter()
            .map(|e| e.trunk.len() + e.heads.len())
            .collect();
        let widths: Vec<usize> = stk.iter().map(|s| s.main_seed().s.ncols()).collect();
        let capture_blocks = stk
            .iter()
            .filter(|s| s.dom.beta_pc.is_empty() && s.unst_rows())
            .count();
        let (stk_passes, report) = {
            let refs: Vec<&EvalSeeded> = stk.iter().map(|s| s.as_ref()).collect();
            bab.eval_main_passes(&refs)
        };
        let own_passes: Vec<Result<PassOut>> = own.iter().map(|s| bab.eval_main_pass(s)).collect();
        let mut mismatches = Vec::new();
        let mut compared = 0usize;
        for (slot, ((((entry, s_stk), s_own), p_stk), p_own)) in entries
            .iter()
            .zip(stk)
            .zip(own)
            .zip(stk_passes)
            .zip(own_passes)
            .enumerate()
        {
            let tag = format!("domain {slot} depth {}", depths[slot]);
            let (p_stk, p_own) = match (p_stk, p_own) {
                (Ok(a), Ok(b)) => (a, b),
                (a, b) => {
                    mismatches.push(format!(
                        "{tag}: pass result stacked_ok={} own_ok={}",
                        a.is_ok(),
                        b.is_ok()
                    ));
                    continue;
                }
            };
            pass_eq(&tag, &p_stk, &p_own, &mut mismatches);
            match (
                bab.eval_finish(entry, s_stk, p_stk),
                bab.eval_finish(entry, s_own, p_own),
            ) {
                (Ok(a), Ok(b)) => eval_eq(&format!("{tag} eval"), &a, &b, &mut mismatches),
                (a, b) => mismatches.push(format!(
                    "{tag}: eval result stacked_ok={} own_ok={}",
                    a.is_ok(),
                    b.is_ok()
                )),
            }
            compared += 1;
        }
        StackedEvalDiff {
            depths,
            widths,
            beta_pc_index,
            beta_blocks,
            capture_blocks,
            report,
            mismatches,
            compared,
        }
    }
}

#[cfg(test)]
mod mr_stack_line_tests {
    use super::*;

    /// `mr_stack_stop_line_has_fixed_schema`: the `[mr-stack]` receipt carries
    /// every field on every line (a fixed schema is what lets a grep tell
    /// "zero" from "this binary lacks B5"), as ONE line, and the census
    /// fields carry the same counts.
    #[test]
    fn mr_stack_stop_line_has_fixed_schema() {
        let line = mr_stack_stop_line();
        assert!(line.starts_with("[mr-stack] main_eval "), "{line}");
        for key in [
            " K=",
            " nf=",
            " blocks=",
            " passes=",
            " batches=",
            " cols_max=",
            " fallback_beta_pc=",
            " fallback_seam=",
            " fallback_err=",
            " solo=",
            " disabled=",
            " engaged=",
            " traj=0x",
            " traj_steps=",
            " traj_trees=",
        ] {
            assert!(line.contains(key), "missing {key}: {line}");
        }
        assert!(!line.contains('\n'), "the receipt must be ONE line");
        let fields = mr_stack_fields();
        for key in [
            " stk_batches=",
            " stk_passes=",
            " stk_blocks=",
            " stk_cols=",
            " stk_k_max=",
            " stk_fallback_beta_pc=",
            " stk_fallback_seam=",
            " stk_fallback_err=",
            " stk_solo=",
            " stk_disabled=",
        ] {
            assert!(fields.contains(key), "missing {key}: {fields}");
        }
    }

    /// `mr_stack_report_absorbs_sums_and_maxima`: counts add, widths take
    /// the maximum (a tree's `K` is its widest stack, not a sum).
    #[test]
    fn mr_stack_report_absorbs_sums_and_maxima() {
        let mut acc = StackReport::default();
        acc.absorb(&StackReport {
            batches: 1,
            passes: 1,
            blocks: 5,
            k_max: 5,
            cols: 7,
            cols_max: 7,
            fallback_beta_pc: 1,
            ..StackReport::default()
        });
        acc.absorb(&StackReport {
            batches: 1,
            passes: 1,
            blocks: 3,
            k_max: 3,
            cols: 9,
            cols_max: 9,
            solo: 1,
            ..StackReport::default()
        });
        assert_eq!(
            acc,
            StackReport {
                batches: 2,
                passes: 2,
                blocks: 8,
                k_max: 5,
                cols: 16,
                cols_max: 9,
                fallback_beta_pc: 1,
                fallback_seam: 0,
                fallback_err: 0,
                solo: 1,
                disabled: 0,
            }
        );
    }

    /// `mr_stack_chunks_respect_the_column_cap`: consecutive, gap-free,
    /// order-preserving cuts; no chunk of two or more exceeds the cap; an item
    /// wider than the cap stands alone; the cap never binds on cifar100-shaped
    /// batches (K = 20 domains of nf <= 3).
    #[test]
    fn mr_stack_chunks_respect_the_column_cap() {
        assert_eq!(stack_chunks(&[], 256), Vec::<std::ops::Range<usize>>::new());
        assert_eq!(stack_chunks(&[3; 20], MAX_STACK_COLS), vec![0..20]);
        assert_eq!(stack_chunks(&[4, 4, 4], 8), vec![0..2, 2..3]);
        assert_eq!(stack_chunks(&[9, 1, 1], 8), vec![0..1, 1..3]);
        assert_eq!(stack_chunks(&[1, 9, 1], 8), vec![0..1, 1..2, 2..3]);
        let widths = [99usize; 20];
        let cuts = stack_chunks(&widths, MAX_STACK_COLS);
        let mut next = 0usize;
        for cut in &cuts {
            assert_eq!(cut.start, next, "gap-free and ordered");
            assert!(cut.end > cut.start, "non-empty");
            let sum: usize = widths[cut.clone()].iter().sum();
            assert!(
                cut.len() == 1 || sum <= MAX_STACK_COLS,
                "{cut:?} sums {sum}"
            );
            next = cut.end;
        }
        assert_eq!(next, widths.len(), "every item covered");
    }

    /// `mr_stack_trajectory_hash_is_bit_sensitive`: the hash separates two
    /// traces that differ in one bit of one value, and is a pure function of
    /// the folded words.
    #[test]
    fn mr_stack_trajectory_hash_is_bit_sensitive() {
        let mut a = B5Trace::default();
        let mut b = B5Trace::default();
        let mut c = B5Trace::default();
        a.f64s(&[0.5, -0.0]);
        b.f64s(&[0.5, -0.0]);
        c.f64s(&[0.5, 0.0]);
        assert_eq!(a.traj, b.traj);
        assert_ne!(a.traj, c.traj, "-0.0 and +0.0 must hash apart");
    }
}
