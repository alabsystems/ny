// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The device wave search's LAUNCH SEQUENCE (#wave-beta-bab, ENGINE_PROGRAM
//! M6), generic over a [`WaveExec`] — the thin layer that owns the device
//! buffers and issues copies and launches. The backend's executor is cudarc
//! (`wave_device.rs`); a standalone harness can supply a raw-driver one and run
//! this very sequence against ny-bbe's CPU reference.
//!
//! One `backward_batch` call cuts its wave into SLABS (at most the upload's
//! `nd` domains and `nl` literals each; a slab's outputs do not depend on the
//! cut) and, per slab, mirrors ny-bbe's `run_domain` for every domain at once:
//! `iters + 1` bound evaluations, each a backward walk (`wave_backward`, op by
//! op in reverse), the concretisation with keep-best / early-stop / last-pass
//! bookkeeping (`wave_concretize`), then — for the domains that step — the
//! linearised forward walk (`wave_forward`) and one projected Adam step
//! (`wave_adam`). The per-domain state comes back after every concretisation,
//! so a slab stops as soon as no domain steps and the deadline is polled
//! between iterations (the plan sizes one iteration to about
//! `PASS_WORK_BUDGET` multiply-adds). Before a slab's first launch every buffer
//! the kernels may index at the slab's shape is checked against what the
//! executor ACTUALLY allocated (the M4 review's `launch_needs` pattern): a
//! launch that could pass an allocation is refused, never issued.

use std::time::Instant;

use ny_core::search::{BranchOutWriter, WaveBatch, WaveOutWriter, WaveSearchCfg};

use super::wave_plan::{
    blocks, kat_problem, kat_wave, launch_needs, oracle_pass, oracle_scores, upload_buffers,
    warm_alpha, warm_beta, AdamConsts, OracleDomain, WaveArgs, WavePlan, WaveProg, DF_BEST_LB,
    DF_BEST_SCALE, DOM_F, DOM_U, DU_EVALS, DU_STATE, F_NIN, F_NOUT, PHASE_DOMAINS, PHASE_LITERALS,
    PHASE_SEED, ST_EARLY, ST_GRAD, ST_IMPROVED, WAVE_BABSR_THREADS, WAVE_MAX_K, WAVE_THREADS,
};

/// A kernel of `kernels/wave_search.cu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    Init,
    Backward,
    Concretize,
    Forward,
    Adam,
    Babsr,
}

impl Kernel {
    /// Entry-point name (the order of `kernels::WAVE_KERNELS`).
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Init => "wave_init",
            Self::Backward => "wave_backward",
            Self::Concretize => "wave_concretize",
            Self::Forward => "wave_forward",
            Self::Adam => "wave_adam",
            Self::Babsr => "wave_babsr",
        }
    }
}

/// The kernel's arguments after [`WaveArgs`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum KArg {
    /// `wave_init`'s phase, `wave_backward` / `wave_forward`'s op.
    Int(i32),
    /// `wave_concretize`'s (last pass, `stop_above`).
    Concretize { last: i32, stop_above: f64 },
    /// `wave_adam`'s constants.
    Adam(AdamConsts),
    /// `wave_babsr` takes nothing else.
    None,
}

/// One launch: grid `(x, y)`, CTA width, the size fields (the executor adds
/// its pointers) and the trailing arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Launch {
    pub(crate) kernel: Kernel,
    pub(crate) grid: (u32, u32),
    pub(crate) block: u32,
    pub(crate) args: WaveArgs,
    pub(crate) arg: KArg,
}

/// One slab's host inputs, in the slab buffers' layout.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct SlabHost {
    pub(crate) rows: Vec<u32>,
    pub(crate) lit_pos: Vec<u32>,
    pub(crate) lit_act: Vec<u32>,
    pub(crate) lit_dom: Vec<u32>,
    /// Raw warm multipliers (the device ingests them).
    pub(crate) beta: Vec<f32>,
    /// Raw warm slopes, `nd x stride`, or empty (every domain starts at the
    /// upload's `alpha0`).
    pub(crate) alpha: Vec<f32>,
}

/// An executor's failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExecError {
    /// A host-side check that never touched the device (a size mismatch, an
    /// unknown buffer, a poisoned lock): the call refuses, the device stays
    /// usable.
    Host(String),
    /// A driver error: the context may be poisoned, so the backend latches.
    Device(String),
}

/// The device side of the sequence. None of its methods may block past its
/// own copy; an `Err` says whether the device was involved ([`ExecError`]).
pub(crate) trait WaveExec {
    /// Elements ACTUALLY allocated for buffer `name` (0 for a name the
    /// executor does not hold, so a check against it fails closed).
    fn len_of(&self, name: &str) -> usize;
    /// Copy a slab's inputs into the slab buffers (from element 0).
    fn put(&mut self, host: &SlabHost) -> Result<(), ExecError>;
    /// Zero the first `n` floats of `coef`.
    fn zero_coef(&mut self, n: usize) -> Result<(), ExecError>;
    /// Issue one launch (the executor fills the pointer fields).
    fn launch(&mut self, l: &Launch) -> Result<(), ExecError>;
    /// Wait for the stream, then copy the first `dom_u.len()` / `dom_f.len()`
    /// elements of `dom_u` / `dom_f` back (an empty slice copies nothing).
    fn get_dom(&mut self, dom_u: &mut [u32], dom_f: &mut [f32]) -> Result<(), ExecError>;
    /// Wait, then copy the first `alpha.len()` / `beta.len()` elements of
    /// `best_alpha` / `best_beta` back.
    fn get_best(&mut self, alpha: &mut [f32], beta: &mut [f32]) -> Result<(), ExecError>;
    /// Wait, then copy the first `pos.len()` / `score.len()` elements of
    /// `bpos` / `bscore` back.
    fn get_branch(&mut self, pos: &mut [u32], score: &mut [f32]) -> Result<(), ExecError>;
}

/// Why a run stopped without a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RunErr {
    /// The deadline passed.
    Deadline,
    /// A host-side refusal that never touched the device (the bounds check
    /// refusing a slab before any launch, a malformed batch the caller let
    /// through, an executor's host check): refuse, never latch.
    Invalid(String),
    /// A device fault (a driver error): the backend latches.
    Fault(String),
}

impl From<ExecError> for RunErr {
    fn from(e: ExecError) -> Self {
        match e {
            ExecError::Host(why) => Self::Invalid(why),
            ExecError::Device(why) => Self::Fault(why),
        }
    }
}

/// Untrusted counters of one call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    /// Bound evaluations over every domain.
    pub(crate) passes: u64,
    /// Domains that left the ascent above `stop_above`.
    pub(crate) early: u64,
    /// Slabs run.
    pub(crate) slabs: u64,
}

fn past(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() > d)
}

/// The end of the slab starting at domain `lo`: at most `plan.nd` domains and
/// `plan.nl` literals, at least one domain (a domain's literals are distinct
/// universe positions, so at most `stride <= plan.nl`; checked anyway).
fn slab_end(batch: &WaveBatch<'_>, lo: usize, plan: &WavePlan) -> Result<usize, RunErr> {
    let base = batch.offsets[lo];
    if (batch.offsets[lo + 1] - base) as usize > plan.nl {
        return Err(RunErr::Invalid(format!(
            "domain {lo} carries more literals than a slab holds ({})",
            plan.nl
        )));
    }
    let mut hi = lo + 1;
    while hi < batch.count
        && hi - lo < plan.nd
        && (batch.offsets[hi + 1] - base) as usize <= plan.nl
    {
        hi += 1;
    }
    Ok(hi)
}

/// The host inputs of domains `lo..hi`.
fn slab_host(batch: &WaveBatch<'_>, lo: usize, hi: usize, stride: usize) -> SlabHost {
    let r = batch.offsets[lo] as usize..batch.offsets[hi] as usize;
    let mut lit_dom = Vec::with_capacity(r.len());
    for d in lo..hi {
        let n = (batch.offsets[d + 1] - batch.offsets[d]) as usize;
        lit_dom.extend(std::iter::repeat_n((d - lo) as u32, n));
    }
    SlabHost {
        rows: batch.rows[lo..hi].to_vec(),
        lit_pos: batch.splits[r.clone()].iter().map(|s| s.pos).collect(),
        lit_act: batch.splits[r.clone()]
            .iter()
            .map(|s| u32::from(s.active))
            .collect(),
        lit_dom,
        beta: batch.beta[r].to_vec(),
        alpha: if batch.alpha.is_empty() {
            Vec::new()
        } else {
            batch.alpha[lo * stride..hi * stride].to_vec()
        },
    }
}

/// Refuse a slab of `nd` domains / `nl` literals whose launches could index
/// past a buffer the executor holds.
fn check_fits<X: WaveExec>(x: &X, prog: &WaveProg, nd: usize, nl: usize) -> Result<(), RunErr> {
    for (name, _, need) in upload_buffers(prog) {
        let have = x.len_of(name);
        if have < need {
            return Err(RunErr::Invalid(format!(
                "upload buffer {name} holds {have} elements, the program needs {need}"
            )));
        }
    }
    for (name, need) in launch_needs(prog.t, prog.stride, nd, nl) {
        let have = x.len_of(name);
        if have < need {
            return Err(RunErr::Invalid(format!(
                "a slab of {nd} domains / {nl} literals needs {need} elements of {name}, which \
                 holds {have}"
            )));
        }
    }
    Ok(())
}

/// A grid dimension, refusing one the driver cannot launch.
fn dim(n: u32) -> Result<u32, RunErr> {
    if n == 0 || n > i32::MAX as u32 {
        return Err(RunErr::Invalid(format!("grid dimension {n} out of range")));
    }
    Ok(n)
}

/// The per-slab launch context.
struct Seq<'p> {
    prog: &'p WaveProg,
    args: WaveArgs,
    nd: usize,
    nl: usize,
}

impl Seq<'_> {
    fn launch<X: WaveExec>(
        &self,
        x: &mut X,
        kernel: Kernel,
        grid: (u32, u32),
        arg: KArg,
    ) -> Result<(), RunErr> {
        let block = if kernel == Kernel::Babsr {
            WAVE_BABSR_THREADS
        } else {
            WAVE_THREADS
        };
        let l = Launch {
            kernel,
            grid: (dim(grid.0)?, dim(grid.1)?),
            block,
            args: self.args,
            arg,
        };
        x.launch(&l).map_err(RunErr::from)
    }

    fn nd_u32(&self) -> u32 {
        u32::try_from(self.nd).unwrap_or(u32::MAX)
    }

    /// `wave_init` phases 0 and 1: warm starts, pins, state.
    fn init<X: WaveExec>(&self, x: &mut X) -> Result<(), RunErr> {
        let s = self.prog.stride;
        let domains = self.nd.saturating_mul(s).saturating_add(self.nd);
        self.launch(
            x,
            Kernel::Init,
            (blocks(domains), 1),
            KArg::Int(PHASE_DOMAINS),
        )?;
        if self.nl > 0 {
            self.launch(
                x,
                Kernel::Init,
                (blocks(self.nl), 1),
                KArg::Int(PHASE_LITERALS),
            )?;
        }
        Ok(())
    }

    /// One backward walk of every active domain: zero, seed, ops in reverse.
    fn backward<X: WaveExec>(&self, x: &mut X) -> Result<(), RunErr> {
        x.zero_coef(self.nd.saturating_mul(self.prog.t))
            .map_err(RunErr::from)?;
        self.launch(x, Kernel::Init, (blocks(self.nd), 1), KArg::Int(PHASE_SEED))?;
        for k in (0..self.prog.nops()).rev() {
            let n = self.prog.op(k)[F_NIN] as usize;
            let grid = blocks(n).saturating_add(1);
            self.launch(
                x,
                Kernel::Backward,
                (grid, self.nd_u32()),
                KArg::Int(k as i32),
            )?;
        }
        Ok(())
    }

    /// One forward walk of every stepping domain.
    fn forward<X: WaveExec>(&self, x: &mut X) -> Result<(), RunErr> {
        for k in 0..self.prog.nops() {
            let n = self.prog.op(k)[F_NOUT] as usize;
            self.launch(
                x,
                Kernel::Forward,
                (blocks(n), self.nd_u32()),
                KArg::Int(k as i32),
            )?;
        }
        Ok(())
    }
}

/// `backward_batch` over `batch` (validated by the caller) into `w`
/// (prepared by the caller to the batch's shape).
pub(crate) fn run_search<X: WaveExec>(
    x: &mut X,
    prog: &WaveProg,
    plan: &WavePlan,
    batch: &WaveBatch<'_>,
    cfg: &WaveSearchCfg,
    w: &mut WaveOutWriter<'_>,
) -> Result<Counts, RunErr> {
    let stride = prog.stride;
    let mut counts = Counts::default();
    let mut lo = 0usize;
    while lo < batch.count {
        if past(batch.deadline) {
            return Err(RunErr::Deadline);
        }
        let hi = slab_end(batch, lo, plan)?;
        let host = slab_host(batch, lo, hi, stride);
        let (nd, nl) = (hi - lo, host.lit_pos.len());
        check_fits(x, prog, nd, nl)?;
        let seq = Seq {
            prog,
            args: WaveArgs::sized(prog, nd, nl, !host.alpha.is_empty(), 1),
            nd,
            nl,
        };
        x.put(&host).map_err(RunErr::from)?;
        seq.init(x)?;
        let mut dom_u = vec![0u32; nd * DOM_U];
        let mut dom_f = vec![0f32; nd * DOM_F];
        let concretize_grid = (blocks(prog.n_in), seq.nd_u32());
        let steps = nd.saturating_mul(stride).saturating_add(nl);
        let mut it = 0usize;
        loop {
            if past(batch.deadline) {
                return Err(RunErr::Deadline);
            }
            seq.backward(x)?;
            let last = it >= cfg.iters;
            seq.launch(
                x,
                Kernel::Concretize,
                concretize_grid,
                KArg::Concretize {
                    last: i32::from(last),
                    stop_above: cfg.stop_above,
                },
            )?;
            // The loop reads only the state words; the bounds come back once,
            // after the slab's last iteration.
            x.get_dom(&mut dom_u, &mut []).map_err(RunErr::from)?;
            let state = |d: usize| dom_u[d * DOM_U + DU_STATE];
            let stepping = (0..nd).filter(|&d| state(d) & ST_GRAD != 0).count();
            let improved = (0..nd).any(|d| state(d) & ST_IMPROVED != 0);
            if stepping > 0 {
                seq.forward(x)?;
            }
            if (stepping > 0 || improved) && steps > 0 {
                seq.launch(
                    x,
                    Kernel::Adam,
                    (blocks(steps), 1),
                    KArg::Adam(AdamConsts::at(it, cfg.lr_alpha, cfg.lr_beta)),
                )?;
            }
            if stepping == 0 || last {
                break;
            }
            it += 1;
        }
        let (a, b) = (
            lo * stride..hi * stride,
            batch.offsets[lo] as usize..batch.offsets[hi] as usize,
        );
        x.get_dom(&mut dom_u, &mut dom_f).map_err(RunErr::from)?;
        x.get_best(&mut w.alpha[a], &mut w.beta[b])
            .map_err(RunErr::from)?;
        for d in 0..nd {
            w.lb[lo + d] = dom_f[d * DOM_F + DF_BEST_LB];
            w.scale[lo + d] = dom_f[d * DOM_F + DF_BEST_SCALE];
            let evals = dom_u[d * DOM_U + DU_EVALS];
            w.evals[lo + d] = evals;
            counts.passes += u64::from(evals);
            counts.early += u64::from(dom_u[d * DOM_U + DU_STATE] & ST_EARLY != 0);
        }
        counts.slabs += 1;
        lo = hi;
    }
    Ok(counts)
}

/// `branch_scores` over `batch` (validated by the caller) with `k` in
/// `1..=WAVE_MAX_K` into `w` (prepared by the caller to `count x k`).
pub(crate) fn run_branch<X: WaveExec>(
    x: &mut X,
    prog: &WaveProg,
    plan: &WavePlan,
    batch: &WaveBatch<'_>,
    k: usize,
    w: &mut BranchOutWriter<'_>,
) -> Result<Counts, RunErr> {
    if k == 0 || k > WAVE_MAX_K {
        return Err(RunErr::Invalid(format!(
            "branch k = {k} outside 1..={WAVE_MAX_K}"
        )));
    }
    let stride = prog.stride;
    let mut counts = Counts::default();
    let mut lo = 0usize;
    while lo < batch.count {
        if past(batch.deadline) {
            return Err(RunErr::Deadline);
        }
        let hi = slab_end(batch, lo, plan)?;
        let host = slab_host(batch, lo, hi, stride);
        let (nd, nl) = (hi - lo, host.lit_pos.len());
        check_fits(x, prog, nd, nl)?;
        let seq = Seq {
            prog,
            args: WaveArgs::sized(prog, nd, nl, !host.alpha.is_empty(), k),
            nd,
            nl,
        };
        x.put(&host).map_err(RunErr::from)?;
        seq.init(x)?;
        seq.backward(x)?;
        let grid = u32::try_from(nd.div_ceil(WAVE_BABSR_THREADS as usize)).unwrap_or(u32::MAX);
        seq.launch(x, Kernel::Babsr, (grid, 1), KArg::None)?;
        x.get_branch(&mut w.pos[lo * k..hi * k], &mut w.score[lo * k..hi * k])
            .map_err(RunErr::from)?;
        counts.slabs += 1;
        lo = hi;
    }
    Ok(counts)
}

// ---------------------------------------------------------------------------
//  The init self-test (G6), on any executor
// ---------------------------------------------------------------------------

/// Same bits, or both NaN, or both zero (a zero's sign is not an output).
#[must_use]
pub(crate) fn same_value(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) || (a == 0.0 && b == 0.0)
}

/// Run the known-answer self-test on an executor holding an upload of
/// `kat_problem(seed)` (`prog`, admitted as `plan`): a bound-only wave whose
/// every domain's bound and absolute-term sum must equal the exact oracle BIT
/// FOR BIT (every sum of the fixture is exact), the warm hints must come back
/// ingested, and the BaBSR candidates must be the oracle's; then a short
/// ascent must stay well formed and keep its best (never below the warm
/// bound). `Err` names the first disagreement.
pub(crate) fn self_test<X: WaveExec>(
    x: &mut X,
    prog: &WaveProg,
    plan: &WavePlan,
    count: usize,
    seed: u64,
) -> Result<(), String> {
    let stride = prog.stride;
    let (rows, offsets, splits, beta, alpha) = kat_wave(stride, count, seed);
    let batch = WaveBatch {
        count,
        rows: &rows,
        offsets: &offsets,
        splits: &splits,
        beta: &beta,
        alpha: &alpha,
        stride,
        deadline: None,
    };
    batch.validate(stride, prog.classes.len())?;
    let run = |x: &mut X, iters: usize| -> Result<ny_core::search::WaveOut, String> {
        let mut out = ny_core::search::WaveOut::default();
        out.prepare(count, stride, splits.len());
        let cfg = WaveSearchCfg {
            iters,
            stop_above: f64::INFINITY,
            ..WaveSearchCfg::default()
        };
        run_search(x, prog, plan, &batch, &cfg, &mut out.writer())
            .map_err(|e| format!("self-test search: {e:?}"))?;
        Ok(out)
    };
    let warm = run(x, 0)?;
    let view = warm.hint().into_ingest();
    let k = 4usize.min(WAVE_MAX_K);
    let mut br = ny_core::search::BranchOut::default();
    br.prepare(count, k);
    run_branch(x, prog, plan, &batch, k, &mut br.writer())
        .map_err(|e| format!("self-test branch: {e:?}"))?;
    let bv = br.hint().into_ingest();
    for d in 0..count {
        let r = offsets[d] as usize..offsets[d + 1] as usize;
        let dom = OracleDomain {
            row: rows[d] as usize,
            lits: splits[r.clone()]
                .iter()
                .zip(&beta[r.clone()])
                .map(|(s, &b)| (s.pos, s.active, warm_beta(b)))
                .collect(),
            alpha: (0..stride)
                .map(|p| warm_alpha(alpha[d * stride + p], prog.fp[prog.sec.alpha0 + p]))
                .collect(),
        };
        let want = oracle_pass::<f64>(prog, &dom);
        let (lb, sc) = (want.lb as f32, want.scale as f32);
        if !same_value(view.lb(d), lb) || !same_value(view.scale(d), sc) || view.evals(d) != 1 {
            return Err(format!(
                "self-test: domain {d} bound (lb, S, evals) = ({:e}, {:e}, {}) on the device vs \
                 the exact ({lb:e}, {sc:e}, 1)",
                view.lb(d),
                view.scale(d),
                view.evals(d)
            ));
        }
        for (g, lit) in r.zip(&dom.lits) {
            if !same_value(view.beta()[g], lit.2) {
                return Err(format!(
                    "self-test: domain {d} literal {g} multiplier not ingested"
                ));
            }
        }
        for (p, &a) in dom.alpha.iter().enumerate() {
            if !same_value(view.alpha(d)[p], a) {
                return Err(format!("self-test: domain {d} slope {p} not ingested"));
            }
        }
        let top = oracle_scores(prog, &dom, &want.lam, k);
        for c in 0..k {
            let (wp, ws) = top.get(c).copied().unwrap_or((u32::MAX, f32::NAN));
            if bv.pos(d)[c] != wp || !same_value(bv.score(d)[c], ws) {
                return Err(format!(
                    "self-test: domain {d} candidate {c} = ({}, {:e}) on the device vs the exact \
                     ({wp}, {ws:e})",
                    bv.pos(d)[c],
                    bv.score(d)[c]
                ));
            }
        }
    }
    let steps = run(x, 3)?;
    let sv = steps.hint().into_ingest();
    if !sv.well_formed() {
        return Err("self-test: the ascent's output is malformed".into());
    }
    for d in 0..count {
        if !matches!(
            sv.lb(d).partial_cmp(&view.lb(d)),
            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
        ) || sv.evals(d) == 0
            || sv.evals(d) > 4
        {
            return Err(format!(
                "self-test: domain {d} ascent kept {:e} (evals {}) below its warm {:e}",
                sv.lb(d),
                sv.evals(d),
                view.lb(d)
            ));
        }
        if !sv.alpha(d).iter().all(|a| (0.0..=1.0).contains(a)) {
            return Err(format!(
                "self-test: domain {d} ascent left a slope outside [0, 1]"
            ));
        }
    }
    if !sv.beta().iter().all(|b| b.is_finite() && *b >= 0.0) {
        return Err("self-test: the ascent left a negative or non-finite multiplier".into());
    }
    Ok(())
}

/// The self-test's problem seed and domain count (shared with the device-free
/// fixture tests).
pub(crate) const KAT_SEED: u64 = 0x5E1F;
/// Domains of the self-test's wave.
pub(crate) const KAT_DOMAINS: usize = 9;

/// The self-test's lowered problem.
pub(crate) fn kat_prog() -> Result<WaveProg, String> {
    WaveProg::lower(&kat_problem(KAT_SEED)).map_err(|e| format!("self-test lowering: {e}"))
}
