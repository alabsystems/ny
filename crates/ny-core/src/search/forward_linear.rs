// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M8a — the lowdim lane's FORWARD-LINEAR search copy (ENGINE_PROGRAM §2 M8,
//! `#m8-input-split-batch`).
//!
//! The `#lowdim-prover` lane bounds each input-split domain by forward-linear
//! intermediates (`d + 1` coefficients per neuron) and a backward-CROWN
//! objective pass. This module is the plain-data copy of the lane's lowered
//! program that an UNTRUSTED backend bounds a whole WAVE of domains with
//! ([`super::SearchBackend::forward_linear_batch`], R1: one call per wave):
//!
//! * [`FlProgram`] — the op list over flat vectors (the lane's `Program`
//!   verbatim, with `u32` sources). `Linear` weights are `Arc`-shared, so a
//!   fold family of 2,260 clauses uploads ONE copy of each weight matrix.
//! * [`FlFamily`] — one structure and, per clause, the constants that differ
//!   between clauses (biases, sparse coefficients and constants, `*Const`
//!   vectors) — `clauses[c][op]`, aligned with `ops`.
//! * [`FlProblem`] — the families plus one table of objective rows; a batch
//!   names, per domain, which rows to bound (a CSR over the table).
//!
//! Every output ([`FlOut`]: per requested row the search lower bound, its
//! absolute-term sum and the bound form's input coefficients) is a HINT. The
//! lane reads it only through [`FlOut::hint`] -> [`super::Hint::into_ingest`]
//! at its one ingest gate, where a hint can only CLAIM a leaf (which the lane
//! then re-bounds with its own certified f64 forms and closes only on the
//! unchanged strict `lb > thr`) or steer a split dimension (any split of a box
//! covers it). THE BACKEND NEVER CERTIFIES.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use super::Hint;

/// One `Linear` op: `y = W x + b`, `W` row-major `n_out x n_in`.
///
/// `w` is SHARED: every program (and every family) that references the same
/// `Arc` references one matrix, and a backend uploads ONE copy per `Arc`
/// (keyed by pointer). The bias is a per-clause constant ([`FlOpConsts::Bias`]).
#[derive(Debug, Clone)]
pub struct FlLinear {
    /// Source op.
    pub src: u32,
    /// Input width (`sizes[src]`).
    pub n_in: u32,
    /// Output width.
    pub n_out: u32,
    /// Row-major `n_out x n_in` weights, exact f64.
    pub w: Arc<[f64]>,
}

/// One op of an [`FlProgram`]. Op `k`'s value is node `k`; op 0 is the input.
#[derive(Debug, Clone)]
pub enum FlOp {
    /// The input vector (op 0 only), width `d`.
    Input,
    /// `W x + b` ([`FlLinear`]; bias per clause).
    Linear(FlLinear),
    /// Per output element `j`: `sum_t coef_t * src_t[idx_t] + k_j`, `k_j` in
    /// `[k_lo_j, k_hi_j]`. `terms[j]` lists `(source op, source element)`;
    /// the coefficients and constants are per clause ([`FlOpConsts::Sparse`],
    /// coefficients in term order, row-major over outputs).
    Sparse {
        /// Per output element, its `(source op, element)` terms.
        terms: Vec<Vec<(u32, u32)>>,
    },
    /// Elementwise ReLU.
    Relu {
        /// Source op.
        src: u32,
    },
    /// Elementwise logistic sigmoid.
    Sigmoid {
        /// Source op.
        src: u32,
    },
    /// The contiguous flat range `[start, start + n)` of the source.
    Slice {
        /// Source op.
        src: u32,
        /// First element.
        start: u32,
    },
    /// Flat concatenation of the sources, in order.
    Concat {
        /// Source ops.
        srcs: Vec<u32>,
    },
    /// The source unchanged.
    Identity {
        /// Source op.
        src: u32,
    },
    /// `a + b` (equal widths).
    Add {
        /// Left operand.
        a: u32,
        /// Right operand.
        b: u32,
    },
    /// `a - b` (equal widths).
    Sub {
        /// Left operand.
        a: u32,
        /// Right operand.
        b: u32,
    },
    /// `v + k` (`k` per clause, [`FlOpConsts::K`]).
    AddConst {
        /// Source op.
        src: u32,
    },
    /// `k - v` (`k` per clause).
    RevSubConst {
        /// Source op.
        src: u32,
    },
    /// `v * k` elementwise (`k` per clause).
    MulConst {
        /// Source op.
        src: u32,
    },
    /// `a * b` elementwise. Expressible; the M8a CPU backend refuses it
    /// (the lane's McCormick planes are not mirrored).
    Mul {
        /// Left operand.
        a: u32,
        /// Right operand.
        b: u32,
    },
    /// `max_j v_j` over the whole source (width 1). Expressible; the M8a CPU
    /// backend refuses it.
    MaxGlobal {
        /// Source op.
        src: u32,
    },
}

impl FlOp {
    /// The ops this op reads.
    #[must_use]
    pub fn sources(&self) -> Vec<u32> {
        match self {
            Self::Input => Vec::new(),
            Self::Linear(l) => vec![l.src],
            Self::Sparse { terms } => {
                let mut s: Vec<u32> = terms.iter().flatten().map(|&(src, _)| src).collect();
                s.sort_unstable();
                s.dedup();
                s
            }
            Self::Relu { src }
            | Self::Sigmoid { src }
            | Self::Slice { src, .. }
            | Self::Identity { src }
            | Self::AddConst { src }
            | Self::RevSubConst { src }
            | Self::MulConst { src }
            | Self::MaxGlobal { src } => vec![*src],
            Self::Concat { srcs } => srcs.clone(),
            Self::Add { a, b } | Self::Sub { a, b } | Self::Mul { a, b } => vec![*a, *b],
        }
    }

    /// Short stable tag (refusal messages).
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Linear(_) => "Linear",
            Self::Sparse { .. } => "Sparse",
            Self::Relu { .. } => "Relu",
            Self::Sigmoid { .. } => "Sigmoid",
            Self::Slice { .. } => "Slice",
            Self::Concat { .. } => "Concat",
            Self::Identity { .. } => "Identity",
            Self::Add { .. } => "Add",
            Self::Sub { .. } => "Sub",
            Self::AddConst { .. } => "AddConst",
            Self::RevSubConst { .. } => "RevSubConst",
            Self::MulConst { .. } => "MulConst",
            Self::Mul { .. } => "Mul",
            Self::MaxGlobal { .. } => "MaxGlobal",
        }
    }
}

/// One program structure: `ops` over flat vectors, `sizes[k]` the width of
/// node `k`, `d = sizes[0]` the input width, `seed` the node the objective
/// rows are over.
#[derive(Debug, Clone)]
pub struct FlProgram {
    /// Input dimension.
    pub d: usize,
    /// Width of every node.
    pub sizes: Vec<usize>,
    /// The objective node.
    pub seed: usize,
    /// Topologically ordered ops (every source strictly earlier).
    pub ops: Vec<FlOp>,
}

/// The per-clause constants of one op.
///
/// The vectors are `Arc`-shared so clauses whose constants coincide (every
/// deeper layer of a fold family) carry one copy; a backend lowers each
/// distinct `Arc` once.
#[derive(Debug, Clone)]
pub enum FlOpConsts {
    /// The op has no constants.
    None,
    /// A `Linear` bias in `[lo, hi]` (width `n_out`).
    Bias {
        /// Lower end.
        lo: Arc<[f64]>,
        /// Upper end.
        hi: Arc<[f64]>,
    },
    /// A `*Const` vector (the op's width).
    K(Arc<[f64]>),
    /// A `Sparse` op's coefficients (one per term, row-major) and constant
    /// interval per output.
    Sparse {
        /// Coefficients, in term order.
        coef: Arc<[f64]>,
        /// Lower constant per output.
        k_lo: Arc<[f64]>,
        /// Upper constant per output.
        k_hi: Arc<[f64]>,
    },
}

/// One structure family: the program and, per clause, its constants
/// (`clauses[c][op]`, aligned with `program.ops`).
#[derive(Debug, Clone)]
pub struct FlFamily {
    /// The shared structure.
    pub program: FlProgram,
    /// Per clause, per op, the constants.
    pub clauses: Vec<Vec<FlOpConsts>>,
}

/// A forward-linear search problem: families plus ONE table of objective
/// rows, each over its family's seed width (every family's seed has the
/// table's width).
#[derive(Debug, Clone)]
pub struct FlProblem {
    /// The families.
    pub families: Vec<FlFamily>,
    /// Objective rows (a batch names rows of this table per domain).
    pub objectives: Vec<Vec<f64>>,
}

fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn program_sizes(p: &FlProgram, f: usize) -> Result<(), String> {
    let n = p.ops.len();
    if n == 0 || p.sizes.len() != n {
        return Err(format!("family {f}: sizes do not match the op list"));
    }
    if p.d == 0 || p.sizes[0] != p.d || !matches!(p.ops[0], FlOp::Input) {
        return Err(format!("family {f}: op 0 must be the width-d input"));
    }
    if p.seed >= n || p.sizes[p.seed] == 0 {
        return Err(format!("family {f}: seed out of range or empty"));
    }
    for (k, op) in p.ops.iter().enumerate().skip(1) {
        let w = p.sizes[k];
        let bad = |why: &str| Err(format!("family {f} op {k} ({}): {why}", op.tag()));
        for s in op.sources() {
            if s as usize >= k {
                return bad("forward reference");
            }
        }
        let size = |s: u32| p.sizes[s as usize];
        match op {
            FlOp::Input => return bad("input after op 0"),
            FlOp::Linear(l) => {
                if size(l.src) != l.n_in as usize || w != l.n_out as usize {
                    return bad("size mismatch");
                }
                if l.w.len() != (l.n_out as usize) * (l.n_in as usize) {
                    return bad("weight length is not n_out x n_in");
                }
                if !finite(&l.w) {
                    return bad("non-finite weight");
                }
            }
            FlOp::Sparse { terms } => {
                if terms.len() != w {
                    return bad("one term list per output");
                }
                for &(s, i) in terms.iter().flatten() {
                    if i as usize >= size(s) {
                        return bad("term element out of range");
                    }
                }
            }
            FlOp::Relu { src }
            | FlOp::Sigmoid { src }
            | FlOp::Identity { src }
            | FlOp::AddConst { src }
            | FlOp::RevSubConst { src }
            | FlOp::MulConst { src } => {
                if size(*src) != w {
                    return bad("size mismatch");
                }
            }
            FlOp::Slice { src, start } => {
                if (*start as usize).saturating_add(w) > size(*src) {
                    return bad("slice past the source");
                }
            }
            FlOp::Concat { srcs } => {
                if srcs.is_empty() || srcs.iter().map(|&s| size(s)).sum::<usize>() != w {
                    return bad("size mismatch");
                }
            }
            FlOp::Add { a, b } | FlOp::Sub { a, b } | FlOp::Mul { a, b } => {
                if size(*a) != w || size(*b) != w {
                    return bad("size mismatch");
                }
            }
            FlOp::MaxGlobal { src } => {
                if w != 1 || size(*src) == 0 {
                    return bad("size mismatch");
                }
            }
        }
    }
    Ok(())
}

fn clause_consts(p: &FlProgram, consts: &[FlOpConsts], f: usize, c: usize) -> Result<(), String> {
    if consts.len() != p.ops.len() {
        return Err(format!(
            "family {f} clause {c}: constants not aligned with ops"
        ));
    }
    for (k, (op, cst)) in p.ops.iter().zip(consts).enumerate() {
        let w = p.sizes[k];
        let bad = |why: &str| Err(format!("family {f} clause {c} op {k}: {why}"));
        match (op, cst) {
            (FlOp::Linear(l), FlOpConsts::Bias { lo, hi }) => {
                let n = l.n_out as usize;
                if lo.len() != n || hi.len() != n {
                    return bad("bias width");
                }
                if !finite(lo) || !finite(hi) || lo.iter().zip(hi.iter()).any(|(a, b)| a > b) {
                    return bad("non-finite or crossed bias");
                }
            }
            (
                FlOp::AddConst { .. } | FlOp::RevSubConst { .. } | FlOp::MulConst { .. },
                FlOpConsts::K(k),
            ) => {
                if k.len() != w || !finite(k) {
                    return bad("constant width or non-finite constant");
                }
            }
            (FlOp::Sparse { terms }, FlOpConsts::Sparse { coef, k_lo, k_hi }) => {
                let n_terms: usize = terms.iter().map(Vec::len).sum();
                if coef.len() != n_terms {
                    return bad("one coefficient per term");
                }
                if k_lo.len() != w || k_hi.len() != w {
                    return bad("constant width");
                }
                if !finite(coef) || !finite(k_lo) || !finite(k_hi) {
                    return bad("non-finite coefficient or constant");
                }
                if k_lo.iter().zip(k_hi.iter()).any(|(a, b)| a > b) {
                    return bad("k_lo > k_hi");
                }
            }
            (
                FlOp::Input
                | FlOp::Relu { .. }
                | FlOp::Sigmoid { .. }
                | FlOp::Slice { .. }
                | FlOp::Concat { .. }
                | FlOp::Identity { .. }
                | FlOp::Add { .. }
                | FlOp::Sub { .. }
                | FlOp::Mul { .. }
                | FlOp::MaxGlobal { .. },
                FlOpConsts::None,
            ) => {}
            _ => return bad("constants of the wrong kind for the op"),
        }
    }
    Ok(())
}

impl FlProblem {
    /// Structural validation. Refuses: no family or no objective; a forward
    /// reference; any size mismatch (Linear, Slice, Concat, Add/Sub/Mul,
    /// unary); `w.len() != n_out * n_in`; non-finite weights, constants or
    /// objectives; clause constants not aligned with the ops or of the wrong
    /// kind; a Sparse coefficient count that differs from its term count, or
    /// `k_lo > k_hi`; an objective whose width is not a family's
    /// `sizes[seed]`.
    pub fn validate(&self) -> Result<(), String> {
        if self.families.is_empty() {
            return Err("forward-linear problem: no family".into());
        }
        if self.objectives.is_empty() {
            return Err("forward-linear problem: no objective row".into());
        }
        for (f, fam) in self.families.iter().enumerate() {
            program_sizes(&fam.program, f)?;
            if fam.clauses.is_empty() {
                return Err(format!("family {f}: no clause"));
            }
            for (c, consts) in fam.clauses.iter().enumerate() {
                clause_consts(&fam.program, consts, f, c)?;
            }
            let width = fam.program.sizes[fam.program.seed];
            if let Some(r) = self.objectives.iter().position(|o| o.len() != width) {
                return Err(format!(
                    "objective {r} has {} entries for family {f}'s width-{width} seed",
                    self.objectives[r].len()
                ));
            }
        }
        if let Some(r) = self.objectives.iter().position(|o| !finite(o)) {
            return Err(format!("objective {r} is not finite"));
        }
        Ok(())
    }

    /// Approximate resident bytes of the problem, counting every shared
    /// `Arc` ONCE (by pointer).
    #[must_use]
    pub fn approx_bytes(&self) -> u64 {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut bytes = 0u64;
        let mut arc = |a: &Arc<[f64]>| {
            if seen.insert(a.as_ptr() as usize) {
                bytes = bytes.saturating_add((a.len() as u64).saturating_mul(8));
            }
        };
        let mut terms = 0u64;
        for fam in &self.families {
            for op in &fam.program.ops {
                match op {
                    FlOp::Linear(l) => arc(&l.w),
                    FlOp::Sparse { terms: t } => {
                        terms = terms.saturating_add(t.iter().map(|v| v.len() as u64).sum());
                    }
                    _ => {}
                }
            }
            for consts in &fam.clauses {
                for cst in consts {
                    match cst {
                        FlOpConsts::None => {}
                        FlOpConsts::Bias { lo, hi } => {
                            arc(lo);
                            arc(hi);
                        }
                        FlOpConsts::K(k) => arc(k),
                        FlOpConsts::Sparse { coef, k_lo, k_hi } => {
                            arc(coef);
                            arc(k_lo);
                            arc(k_hi);
                        }
                    }
                }
            }
        }
        let objs: u64 = self
            .objectives
            .iter()
            .map(|o| (o.len() as u64).saturating_mul(8))
            .fold(0, u64::saturating_add);
        bytes
            .saturating_add(terms.saturating_mul(8))
            .saturating_add(objs)
    }
}

/// A backend's upload of an [`FlProblem`]. Opaque: only the backend that made
/// it can read its payload, and any other backend refuses it. It remembers
/// the problem's shape (per family its input width and clause count, and the
/// objective table's length) so a batch can be validated against it.
pub struct FlHandle {
    backend: &'static str,
    families: Vec<(usize, usize)>,
    objectives: usize,
    payload: Box<dyn std::any::Any + Send + Sync>,
}

impl FlHandle {
    /// Wrap a backend's upload of `problem`.
    pub fn new<T: std::any::Any + Send + Sync>(
        backend: &'static str,
        problem: &FlProblem,
        payload: T,
    ) -> Self {
        Self {
            backend,
            families: problem
                .families
                .iter()
                .map(|f| (f.program.d, f.clauses.len()))
                .collect(),
            objectives: problem.objectives.len(),
            payload: Box::new(payload),
        }
    }

    /// The backend that made this handle.
    #[must_use]
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Families of the uploaded problem.
    #[must_use]
    pub fn families(&self) -> usize {
        self.families.len()
    }

    /// Family `f`'s `(input width, clause count)`.
    #[must_use]
    pub fn family(&self, f: usize) -> Option<(usize, usize)> {
        self.families.get(f).copied()
    }

    /// Rows of the uploaded objective table.
    #[must_use]
    pub fn objectives(&self) -> usize {
        self.objectives
    }

    /// The payload, if it is a `T`.
    #[must_use]
    pub fn payload<T: std::any::Any>(&self) -> Option<&T> {
        self.payload.downcast_ref::<T>()
    }
}

impl fmt::Debug for FlHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlHandle")
            .field("backend", &self.backend)
            .field("families", &self.families)
            .field("objectives", &self.objectives)
            .finish_non_exhaustive()
    }
}

/// One wave of domains of ONE family. Domain `b` is the box
/// `[lo[b*d..], hi[b*d..]]` of clause `clause[b]` (clause 0 when `clause` is
/// empty) and asks for the objective rows `obj_rows[obj_offsets[b] ..
/// obj_offsets[b + 1]]` (a CSR; row `r` of the output is CSR position `r`).
#[derive(Debug, Clone, Copy)]
pub struct FlBatch<'a> {
    /// The family (an index into [`FlProblem::families`]).
    pub family: u32,
    /// Domains.
    pub count: usize,
    /// Input width (must equal the family's).
    pub d: usize,
    /// `count x d` lower corners.
    pub lo: &'a [f64],
    /// `count x d` upper corners.
    pub hi: &'a [f64],
    /// Per domain its clause, or empty (every domain is clause 0).
    pub clause: &'a [u32],
    /// `count + 1` non-decreasing offsets into `obj_rows`, from 0.
    pub obj_offsets: &'a [u32],
    /// Objective-table rows, concatenated per domain.
    pub obj_rows: &'a [u32],
    /// Stop (refusing with [`super::Refusal::Deadline`]) once this passes.
    pub deadline: Option<Instant>,
}

impl FlBatch<'_> {
    /// Shape validation against an upload. Refuses: a family out of range; a
    /// `d` that differs from the family's; boxes of the wrong length, crossed
    /// or non-finite; a clause list of the wrong length or a clause out of
    /// range; offsets that are not a partition of `obj_rows`; a domain
    /// without objective rows; a row out of the table.
    pub fn validate(&self, h: &FlHandle) -> Result<(), String> {
        let Some((d, clauses)) = h.family(self.family as usize) else {
            return Err("forward-linear batch: family out of range".into());
        };
        if self.d != d {
            return Err("forward-linear batch: d differs from the family's".into());
        }
        let len = self.count.checked_mul(d);
        if len != Some(self.lo.len()) || len != Some(self.hi.len()) {
            return Err("forward-linear batch: boxes are not count x d".into());
        }
        if self
            .lo
            .iter()
            .zip(self.hi)
            .any(|(l, u)| !(l.is_finite() && u.is_finite() && l <= u))
        {
            return Err("forward-linear batch: a box is crossed or non-finite".into());
        }
        if !self.clause.is_empty() && self.clause.len() != self.count {
            return Err("forward-linear batch: clause list is neither empty nor count".into());
        }
        if self.clause.iter().any(|&c| c as usize >= clauses) {
            return Err("forward-linear batch: clause out of range".into());
        }
        if self.obj_offsets.len() != self.count + 1
            || self.obj_offsets.first() != Some(&0)
            || self.obj_offsets.windows(2).any(|w| w[0] > w[1])
            || self.obj_offsets.last().map(|&o| o as usize) != Some(self.obj_rows.len())
        {
            return Err("forward-linear batch: offsets are not a partition of the rows".into());
        }
        if self.obj_offsets.windows(2).any(|w| w[0] == w[1]) {
            return Err("forward-linear batch: a domain without objective rows".into());
        }
        if self.obj_rows.iter().any(|&r| r as usize >= h.objectives()) {
            return Err("forward-linear batch: objective row out of range".into());
        }
        Ok(())
    }
}

/// The per-row outputs of [`super::SearchBackend::forward_linear_batch`]:
/// buffers the BACKEND fills (through [`FlOut::writer`]) and the lane reads
/// only through [`FlOut::hint`] (its ingest gate).
///
/// Per CSR row `r`: the search lower bound `lb` (f32; NaN = none), `scale`,
/// the sum of the absolute terms of that bound (the f32 error normaliser),
/// and `grad`, the bound form's `d` input coefficients (the split hint).
#[derive(Debug, Clone, Default)]
pub struct FlOut {
    rows: usize,
    d: usize,
    lb: Vec<f32>,
    scale: Vec<f32>,
    grad: Vec<f32>,
}

/// Mutable view of an [`FlOut`] for the backend that fills it.
#[derive(Debug)]
pub struct FlOutWriter<'a> {
    /// CSR rows.
    pub rows: usize,
    /// Input width.
    pub d: usize,
    /// Search lower bound per row.
    pub lb: &'a mut [f32],
    /// Absolute-term sum per row.
    pub scale: &'a mut [f32],
    /// `rows x d` input coefficients.
    pub grad: &'a mut [f32],
}

/// Read-only view of an [`FlOut`], reached only through
/// [`super::Hint::into_ingest`]. Every accessor returns search values.
#[derive(Debug, Clone, Copy)]
pub struct FlView<'a> {
    rows: usize,
    d: usize,
    lb: &'a [f32],
    scale: &'a [f32],
    grad: &'a [f32],
}

impl FlOut {
    /// Resize to `rows` rows of `d` coefficients, every value NaN.
    pub fn prepare(&mut self, rows: usize, d: usize) {
        self.rows = rows;
        self.d = d;
        for (buf, len) in [
            (&mut self.lb, rows),
            (&mut self.scale, rows),
            (&mut self.grad, rows.saturating_mul(d)),
        ] {
            buf.clear();
            buf.resize(len, f32::NAN);
        }
    }

    /// The backend's mutable view.
    pub fn writer(&mut self) -> FlOutWriter<'_> {
        FlOutWriter {
            rows: self.rows,
            d: self.d,
            lb: &mut self.lb,
            scale: &mut self.scale,
            grad: &mut self.grad,
        }
    }

    /// The lane's view, as a hint for its ingest gate.
    pub fn hint(&self) -> Hint<FlView<'_>> {
        Hint(FlView {
            rows: self.rows,
            d: self.d,
            lb: &self.lb,
            scale: &self.scale,
            grad: &self.grad,
        })
    }

    /// Bytes [`Self::prepare`] allocates.
    #[must_use]
    pub fn bytes_for(rows: usize, d: usize) -> u64 {
        (rows as u64).saturating_mul(8u64.saturating_add((d as u64).saturating_mul(4)))
    }
}

impl<'a> FlView<'a> {
    /// CSR rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Input width.
    #[must_use]
    pub fn d(&self) -> usize {
        self.d
    }

    /// True iff every buffer has the shape `rows` and `d` promise (a backend
    /// could have written an `FlOut` of another shape).
    #[must_use]
    pub fn well_formed(&self) -> bool {
        self.lb.len() == self.rows
            && self.scale.len() == self.rows
            && Some(self.grad.len()) == self.rows.checked_mul(self.d)
    }

    /// Row `r`'s search lower bound (NaN = none).
    #[must_use]
    pub fn lb(&self, r: usize) -> f32 {
        self.lb[r]
    }

    /// Row `r`'s absolute-term sum.
    #[must_use]
    pub fn scale(&self, r: usize) -> f32 {
        self.scale[r]
    }

    /// Row `r`'s input coefficients (`d` values).
    #[must_use]
    pub fn grad(&self, r: usize) -> &'a [f32] {
        &self.grad[r * self.d..(r + 1) * self.d]
    }
}

/// Telemetry of one [`super::SearchBackend::forward_linear_batch`] call
/// (untrusted).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FlStats {
    /// Domains searched.
    pub domains: u64,
    /// CSR rows bounded.
    pub rows: u64,
    /// Wall seconds.
    pub secs: f64,
    /// Worker threads used.
    pub threads: usize,
    /// Domains per SIMD block.
    pub max_block: usize,
}

#[cfg(test)]
mod tests {
    use super::super::{BackendCaps, DeviceLease, Refusal, SearchBackend, SearchPrecision};
    use super::*;

    fn arc(v: &[f64]) -> Arc<[f64]> {
        v.iter().copied().collect()
    }

    /// `x (2) -> Linear 3 -> Relu -> Linear 1`, one clause.
    fn program() -> FlFamily {
        let w1 = arc(&[1.0, -1.0, 0.5, 2.0, -0.25, 0.75]);
        let w2 = arc(&[1.0, -2.0, 0.5]);
        FlFamily {
            program: FlProgram {
                d: 2,
                sizes: vec![2, 3, 3, 1],
                seed: 3,
                ops: vec![
                    FlOp::Input,
                    FlOp::Linear(FlLinear {
                        src: 0,
                        n_in: 2,
                        n_out: 3,
                        w: w1,
                    }),
                    FlOp::Relu { src: 1 },
                    FlOp::Linear(FlLinear {
                        src: 2,
                        n_in: 3,
                        n_out: 1,
                        w: w2,
                    }),
                ],
            },
            clauses: vec![vec![
                FlOpConsts::None,
                FlOpConsts::Bias {
                    lo: arc(&[0.1, -0.2, 0.0]),
                    hi: arc(&[0.1, -0.2, 0.0]),
                },
                FlOpConsts::None,
                FlOpConsts::Bias {
                    lo: arc(&[0.5]),
                    hi: arc(&[0.5]),
                },
            ]],
        }
    }

    fn problem() -> FlProblem {
        FlProblem {
            families: vec![program()],
            objectives: vec![vec![1.0], vec![-1.0]],
        }
    }

    #[test]
    fn fl_problem_validation_refuses_bad_ops_shapes_consts_and_objectives() {
        assert!(problem().validate().is_ok());
        assert!(problem().approx_bytes() > 0);

        let mut p = problem();
        p.families.clear();
        assert!(p.validate().is_err(), "no family");
        let mut p = problem();
        p.objectives.clear();
        assert!(p.validate().is_err(), "no objective");

        // Forward reference.
        let mut p = problem();
        p.families[0].program.ops[2] = FlOp::Relu { src: 3 };
        assert!(p.validate().is_err(), "a forward reference must refuse");
        // Input after op 0.
        let mut p = problem();
        p.families[0].program.ops[2] = FlOp::Input;
        assert!(p.validate().is_err(), "an input after op 0 must refuse");
        // Linear shape.
        let mut p = problem();
        if let FlOp::Linear(l) = &mut p.families[0].program.ops[1] {
            l.w = arc(&[1.0, 2.0]);
        }
        assert!(p.validate().is_err(), "w.len() != n_out x n_in must refuse");
        let mut p = problem();
        if let FlOp::Linear(l) = &mut p.families[0].program.ops[1] {
            l.n_in = 3;
        }
        assert!(p.validate().is_err(), "n_in != sizes[src] must refuse");
        let mut p = problem();
        if let FlOp::Linear(l) = &mut p.families[0].program.ops[3] {
            l.w = arc(&[1.0, f64::NAN, 0.5]);
        }
        assert!(p.validate().is_err(), "a non-finite weight must refuse");
        // Unary / Slice / Concat / Add sizes.
        let mut p = problem();
        p.families[0].program.sizes[2] = 2;
        assert!(p.validate().is_err(), "a unary size mismatch must refuse");
        let mut p = problem();
        p.families[0].program.ops[2] = FlOp::Slice { src: 1, start: 2 };
        assert!(p.validate().is_err(), "a slice past its source must refuse");
        let mut p = problem();
        p.families[0].program.ops[2] = FlOp::Concat { srcs: vec![0] };
        assert!(
            p.validate().is_err(),
            "a concat of the wrong width must refuse"
        );
        let mut p = problem();
        p.families[0].program.ops[2] = FlOp::Add { a: 1, b: 0 };
        assert!(
            p.validate().is_err(),
            "an add of unequal widths must refuse"
        );
        // Constants.
        let mut p = problem();
        p.families[0].clauses[0].pop();
        assert!(p.validate().is_err(), "constants not aligned with ops");
        let mut p = problem();
        p.families[0].clauses[0][1] = FlOpConsts::None;
        assert!(p.validate().is_err(), "a Linear without its bias");
        let mut p = problem();
        p.families[0].clauses[0][2] = FlOpConsts::K(arc(&[1.0, 1.0, 1.0]));
        assert!(p.validate().is_err(), "a Relu with a constant");
        let mut p = problem();
        p.families[0].clauses[0][1] = FlOpConsts::Bias {
            lo: arc(&[0.2, 0.0, 0.0]),
            hi: arc(&[0.1, 0.0, 0.0]),
        };
        assert!(p.validate().is_err(), "a crossed bias must refuse");
        let mut p = problem();
        p.families[0].clauses[0][1] = FlOpConsts::Bias {
            lo: arc(&[0.0, f64::INFINITY, 0.0]),
            hi: arc(&[0.0, f64::INFINITY, 0.0]),
        };
        assert!(p.validate().is_err(), "a non-finite bias must refuse");
        // Sparse coefficient counts and k_lo > k_hi.
        let sparse = |coef: &[f64], k_lo: f64, k_hi: f64| {
            let mut p = problem();
            p.families[0].program.ops[2] = FlOp::Sparse {
                terms: vec![vec![(1, 0), (1, 2)], vec![(0, 1)], vec![]],
            };
            p.families[0].clauses[0][2] = FlOpConsts::Sparse {
                coef: arc(coef),
                k_lo: arc(&[k_lo; 3]),
                k_hi: arc(&[k_hi; 3]),
            };
            p
        };
        assert!(sparse(&[1.0, -1.0, 2.0], 0.0, 0.5).validate().is_ok());
        assert!(
            sparse(&[1.0, -1.0], 0.0, 0.5).validate().is_err(),
            "coefficient count != term count must refuse"
        );
        assert!(
            sparse(&[1.0, -1.0, 2.0], 0.5, 0.0).validate().is_err(),
            "k_lo > k_hi must refuse"
        );
        let mut p = sparse(&[1.0, -1.0, 2.0], 0.0, 0.5);
        p.families[0].program.ops[2] = FlOp::Sparse {
            terms: vec![vec![(1, 7), (1, 2)], vec![(0, 1)], vec![]],
        };
        assert!(p.validate().is_err(), "a term element out of range");
        // Objectives.
        let mut p = problem();
        p.objectives.push(vec![1.0, 2.0]);
        assert!(p.validate().is_err(), "an objective of the wrong width");
        let mut p = problem();
        p.objectives[1][0] = f64::NAN;
        assert!(p.validate().is_err(), "a non-finite objective");
    }

    #[test]
    fn fl_problem_counts_a_shared_arc_once() {
        let one = problem();
        let mut two = problem();
        two.families.push(program());
        // The second family's weights are different Arcs: more bytes.
        assert!(two.approx_bytes() > one.approx_bytes());
        let mut shared = problem();
        shared.families.push(shared.families[0].clone());
        // A clone shares every Arc: only the op list's terms and objectives count.
        assert_eq!(shared.approx_bytes(), one.approx_bytes());
    }

    #[test]
    fn fl_batch_validation_refuses_bad_family_boxes_clauses_and_csr() {
        let h = FlHandle::new("test", &problem(), ());
        let lo = [0.0, -1.0, 0.25, 0.0];
        let hi = [1.0, 1.0, 0.5, 0.0];
        let off = [0u32, 1, 3];
        let rows = [0u32, 0, 1];
        let ok = FlBatch {
            family: 0,
            count: 2,
            d: 2,
            lo: &lo,
            hi: &hi,
            clause: &[],
            obj_offsets: &off,
            obj_rows: &rows,
            deadline: None,
        };
        assert!(ok.validate(&h).is_ok());
        assert!(
            FlBatch { family: 1, ..ok }.validate(&h).is_err(),
            "bad family"
        );
        assert!(FlBatch { d: 3, ..ok }.validate(&h).is_err(), "d mismatch");
        assert!(
            FlBatch { count: 1, ..ok }.validate(&h).is_err(),
            "box length"
        );
        let crossed = [0.0, 2.0, 0.25, 0.0];
        assert!(
            FlBatch { lo: &crossed, ..ok }.validate(&h).is_err(),
            "a crossed box"
        );
        let nan = [0.0, f64::NAN, 0.25, 0.0];
        assert!(
            FlBatch { lo: &nan, ..ok }.validate(&h).is_err(),
            "a NaN box"
        );
        let one = [0u32];
        assert!(
            FlBatch { clause: &one, ..ok }.validate(&h).is_err(),
            "clause list of the wrong length"
        );
        let far = [0u32, 1];
        assert!(
            FlBatch { clause: &far, ..ok }.validate(&h).is_err(),
            "clause out of range"
        );
        let ok_clause = [0u32, 0];
        assert!(FlBatch {
            clause: &ok_clause,
            ..ok
        }
        .validate(&h)
        .is_ok());
        let not_partition = [0u32, 1, 2];
        assert!(
            FlBatch {
                obj_offsets: &not_partition,
                ..ok
            }
            .validate(&h)
            .is_err(),
            "offsets must end at the row count"
        );
        let decreasing = [0u32, 3, 3];
        assert!(
            FlBatch {
                obj_offsets: &decreasing,
                ..ok
            }
            .validate(&h)
            .is_err(),
            "a domain without rows"
        );
        let bad_rows = [0u32, 0, 2];
        assert!(
            FlBatch {
                obj_rows: &bad_rows,
                ..ok
            }
            .validate(&h)
            .is_err(),
            "a row out of the table"
        );
    }

    #[test]
    fn fl_out_shapes_views_and_handles() {
        let mut out = FlOut::default();
        out.prepare(3, 2);
        {
            let w = out.writer();
            assert_eq!((w.rows, w.d), (3, 2));
            assert_eq!(w.grad.len(), 6);
            w.lb[1] = 0.25;
            w.scale[1] = 2.0;
            w.grad[3] = -1.5;
        }
        let view = out.hint().into_ingest();
        assert!(view.well_formed());
        assert_eq!((view.rows(), view.d()), (3, 2));
        assert_eq!(view.lb(1), 0.25);
        assert_eq!(view.scale(1), 2.0);
        assert!(view.lb(0).is_nan(), "unwritten bounds are NaN");
        assert!(view.grad(0).iter().all(|g| g.is_nan()));
        assert_eq!(view.grad(1)[1], -1.5);
        assert_eq!(FlOut::bytes_for(3, 2), 3 * (8 + 8));
        let h = FlHandle::new("test", &problem(), 7u32);
        assert_eq!(h.backend(), "test");
        assert_eq!(h.families(), 1);
        assert_eq!(h.family(0), Some((2, 1)));
        assert_eq!(h.family(1), None);
        assert_eq!(h.objectives(), 2);
        assert_eq!(h.payload::<u32>(), Some(&7));
        assert!(h.payload::<u64>().is_none());
        assert!(format!("{h:?}").contains("objectives: 2"));
    }

    struct Bare;
    impl SearchBackend for Bare {
        fn name(&self) -> &'static str {
            "bare"
        }
        fn caps(&self) -> BackendCaps {
            BackendCaps::Cpu {
                threads: 1,
                lanes: 1,
                precision: SearchPrecision::F64,
            }
        }
    }

    #[test]
    fn fl_default_backend_refuses_both_methods_typed() {
        let b = Bare;
        let lease = DeviceLease { bytes: 1 << 30 };
        assert_eq!(
            b.upload_forward_linear(&problem(), &lease).err(),
            Some(Refusal::Unsupported("upload_forward_linear"))
        );
        let h = FlHandle::new("bare", &problem(), ());
        let lo = [0.0, 0.0];
        let off = [0u32, 1];
        let rows = [0u32];
        let batch = FlBatch {
            family: 0,
            count: 1,
            d: 2,
            lo: &lo,
            hi: &lo,
            clause: &[],
            obj_offsets: &off,
            obj_rows: &rows,
            deadline: None,
        };
        let mut out = FlOut::default();
        assert_eq!(
            b.forward_linear_batch(&h, &batch, &mut out).err(),
            Some(Refusal::Unsupported("forward_linear_batch"))
        );
    }
}
