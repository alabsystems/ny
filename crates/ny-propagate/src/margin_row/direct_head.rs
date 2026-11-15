// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Direct all-head scalar source, before identity-row production. This has no
//! YPack or affine-envelope authority. Frozen trunk boxes enclose Gemm1 Y;
//! every advised point functional is independently replayed through the CPU
//! engine with all authored Q/intercepts and arithmetic error retained.

use super::super::bounds::YBox;
use super::super::head_group_source::RootFunctionalSource;
use super::*;
use ny_core::head_functional_advice::{
    HeadFunctionalLpHint, HeadFunctionalLpRequest, HeadFunctionalLpRow, FUNCTIONAL_MAX_FEATURES,
    FUNCTIONAL_PROTOCOL_VERSION,
};
use ny_core::head_phase_advice::HeadPhaseAdvice;
use std::sync::Arc;

const MAX_SOURCE_TERMS: usize = 67_108_864;
// This direct-only batch serves every admitted original margin (n_out <= 10).
// Existing selected-group/post-pack batches retain their four-column cap.
const MAX_DIRECT_NATIVE_COLUMNS: usize = 9;

#[path = "adaptive_direct_head.rs"]
mod adaptive;
#[path = "transport_direct_head.rs"]
mod transport;

/// Private producer output. None of its numerical fields is deserializable or
/// mutable through the driver API; the seal binds the original class mapping.
struct DirectScalarHeadSource {
    seal: RootFunctionalSource,
    root: Vec<I>,
    features: Vec<I>,
    y: Vec<I>,
    target: usize,
    adversaries: Vec<usize>,
}

impl DirectScalarHeadSource {
    fn build(
        eng: &BackwardEngine<'_>,
        target: usize,
        adversaries: &[usize],
        deadline: Instant,
    ) -> Option<Self> {
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        let seal = RootFunctionalSource::capture(eng, target, adversaries)?;
        let (input, weights, bias, bias_err, rel, (heads, width)) =
            match &eng.net.ops[eng.net.i_gemm1] {
                TwinOp::Gemm {
                    input,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err,
                    shape,
                } => (*input, weight, bias, bias_err, *weight_rel_err, *shape),
                _ => return None,
            };
        if !(2..=MAX_HEAD).contains(&heads)
            || width == 0
            || heads != eng.net.n_y
            || heads.checked_mul(width)? > MAX_SOURCE_TERMS
            || weights.len() != heads.checked_mul(width)?
            || bias.len() != heads
            || bias_err.len() != heads
            || !good(rel)
            || rel < 0.0
        {
            return None;
        }
        // The admitted tail is exactly Gemm1 -> head ReLU -> final Gemm.
        // Gemm1's input must be the final frozen trunk ReLU through aliases.
        let (final_weights, final_bias, final_shape) = eng.net.gemm2();
        if eng.net.ops.len() != eng.net.i_gemm1.checked_add(3)?
            || !matches!(&eng.net.ops[eng.net.i_gemm1 + 1],
                TwinOp::Relu { input, layer } if *input == eng.net.i_gemm1 + 1 && *layer == usize::MAX)
            || !matches!(&eng.net.ops[eng.net.i_gemm1 + 2],
                TwinOp::Gemm { input, weight_rel_err, bias_err, .. }
                if *input == eng.net.i_gemm1 + 2 && *weight_rel_err == 0.0
                    && bias_err.iter().all(|e| *e == 0.0))
            || final_shape != (eng.net.n_out, heads)
            || final_weights.len() != eng.net.n_out.checked_mul(heads)?
            || final_bias.len() != eng.net.n_out
        {
            return None;
        }
        let mut tensor = input;
        while tensor > 0 {
            if !live(deadline) {
                return None;
            }
            match &eng.net.ops[tensor - 1] {
                TwinOp::Flatten { input } => tensor = *input,
                _ => break,
            }
        }
        let op = tensor.checked_sub(1)?;
        if !matches!(&eng.net.ops[op], TwinOp::Relu { layer, .. } if *layer != usize::MAX) {
            return None;
        }
        let layer = eng
            .root
            .layers
            .iter()
            .find(|l| l.op == op && l.n == width)?;
        if layer.l.len() != width || layer.u.len() != width {
            return None;
        }
        let mut features = Vec::with_capacity(width);
        for i in 0..width {
            if i & 63 == 0 && !live(deadline) {
                return None;
            }
            let x = I {
                lo: layer.l[i].max(0.0),
                hi: layer.u[i].max(0.0),
            };
            if !x.valid() {
                return None;
            }
            features.push(x);
        }
        let mut y = Vec::with_capacity(heads);
        for j in 0..heads {
            if !live(deadline) || !good(bias_err[j]) || bias_err[j] < 0.0 {
                return None;
            }
            let mut row = I::p(bias[j]).add(I {
                lo: -bias_err[j],
                hi: bias_err[j],
            })?;
            for (k, x) in features.iter().enumerate() {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                let w = weights[j * width + k];
                // The fold ball is |W-W'| <= rel*|W'|. Directed multiplication
                // prices even a product rounded to zero before amplification.
                let dust = I::p(rel).mul(I::p(w.abs()))?.hi;
                let coefficient = I::p(w).add(I {
                    lo: -dust,
                    hi: dust,
                })?;
                row = row.add(coefficient.mul(*x)?)?;
            }
            y.push(row);
        }
        let root: Vec<_> = eng
            .root
            .lo
            .iter()
            .zip(&eng.root.hi)
            .map(|(&lo, &hi)| I { lo, hi })
            .collect();
        if root.len() != eng.net.n_in
            || root.iter().any(|x| !x.valid())
            || !live(deadline)
            || !fp_qualified()
            || !seal.same_frame(eng, target, adversaries)
            || !live(deadline)
        {
            return None;
        }
        Some(Self {
            seal,
            root,
            features,
            y,
            target,
            adversaries: adversaries.to_vec(),
        })
    }

    fn margin(
        &self,
        eng: &BackwardEngine<'_>,
        row: usize,
        deadline: Instant,
    ) -> Option<(Vec<I>, I)> {
        let adversary = *self.adversaries.get(row)?;
        let (w, b, (_, width)) = eng.net.gemm2();
        if width != self.y.len() || !live(deadline) {
            return None;
        }
        let mut q = Vec::with_capacity(width);
        for j in 0..width {
            if j & 63 == 0 && !live(deadline) {
                return None;
            }
            q.push(I::p(w[self.target * width + j]).sub(I::p(w[adversary * width + j]))?);
        }
        Some((q, I::p(b[self.target]).sub(I::p(b[adversary]))?))
    }

    fn baseline(&self, q: &[I], mut b: I, deadline: Instant) -> Option<f64> {
        if q.len() != self.y.len() {
            return None;
        }
        for (&q, &y) in q.iter().zip(&self.y) {
            if !live(deadline) {
                return None;
            }
            b = b.add(q.mul(I {
                lo: y.lo.max(0.0),
                hi: y.hi.max(0.0),
            })?)?;
        }
        Some(b.lo)
    }
}

/// A complete baseline exists for every original margin before any provider
/// is constructed. A partial prelude can be MAX-merged only into that mapping.
pub(in crate::margin_row) struct DirectPrelude {
    source: DirectScalarHeadSource,
    bounds: Vec<f64>,
}

/// Travels only with the original outer ledger. The common authorizer checks
/// it after containment and hint-stream work, including mixed root/tree covers.
pub(in crate::margin_row) struct DirectPublicationGuard {
    seal: RootFunctionalSource,
    deadline: Instant,
}
impl DirectPublicationGuard {
    pub(in crate::margin_row) fn valid(
        &self,
        eng: &BackwardEngine<'_>,
        target: usize,
        adversaries: &[usize],
    ) -> bool {
        live(self.deadline)
            && fp_qualified()
            && self.seal.matches(eng, target, adversaries)
            && live(self.deadline)
    }
    #[cfg(test)]
    pub(in crate::margin_row) fn expire_for_test(&mut self) {
        self.deadline = Instant::now();
    }
}

impl DirectPrelude {
    pub(in crate::margin_row) fn matches(
        &self,
        eng: &BackwardEngine<'_>,
        target: usize,
        adversaries: &[usize],
        deadline: Instant,
    ) -> bool {
        live(deadline)
            && fp_qualified()
            && self.bounds.len() == adversaries.len()
            && self.bounds.iter().all(|b| good(*b))
            && self.source.seal.matches(eng, target, adversaries)
            && live(deadline)
    }
    pub(in crate::margin_row) fn bounds(&self) -> &[f64] {
        &self.bounds
    }
    pub(in crate::margin_row) fn complete(&self) -> bool {
        !self.bounds.is_empty() && self.bounds.iter().all(|b| good(*b) && *b > 0.0)
    }
    pub(in crate::margin_row) fn ybox(&self) -> YBox {
        YBox {
            ly: self.source.y.iter().map(|y| y.lo).collect(),
            uy: self.source.y.iter().map(|y| y.hi).collect(),
        }
    }
    pub(in crate::margin_row) fn publication_guard(
        &self,
        deadline: Instant,
    ) -> DirectPublicationGuard {
        DirectPublicationGuard {
            seal: self.source.seal.clone(),
            deadline,
        }
    }
}

pub(in crate::margin_row) struct DirectAttempt {
    pub(in crate::margin_row) prelude: Option<DirectPrelude>,
    budget: advice::AdviceBudget,
    calls: usize,
    columns: usize,
    passes: usize,
    elapsed: Duration,
    adaptive: adaptive::Report,
    transport: transport::Report,
}
impl DirectAttempt {
    /// Root owners may replace or update the gate frame. Never transfer old
    /// numeric bounds across that boundary, even if the new bytes coincide;
    /// the already spent compute/proposal allowance remains this invocation's.
    pub(in crate::margin_row) fn retain_current_frame(
        &mut self,
        eng: &BackwardEngine<'_>,
        target: usize,
        adversaries: &[usize],
        deadline: Option<Instant>,
    ) {
        let begin = Instant::now();
        let valid = self.prelude.as_ref().is_some_and(|prelude| {
            deadline.is_some_and(|deadline| prelude.matches(eng, target, adversaries, deadline))
        });
        if !valid {
            self.prelude = None;
        }
        self.charge_compute(begin.elapsed());
    }

    pub(in crate::margin_row) fn refine_existing(
        &mut self,
        eng: &BackwardEngine<'_>,
        pack: &YPack,
        target: usize,
        adversaries: &[usize],
        existing: &[f64],
        policy: HeadGroupPolicy,
        deadline: Instant,
    ) -> Option<HeadGroupReport> {
        // The mandatory ordinary root pass is not optional compute. Only
        // elapsed work actually spent by this prepass is subtracted here.
        let begin = Instant::now();
        let result = (|| {
            let remaining = policy.max_seconds() - self.elapsed.as_secs_f64();
            let policy = HeadGroupPolicy::new(
                remaining,
                policy.groups_per_margin(),
                policy.proposal_rounds(),
            )?;
            improve_root_margins_with_budget(
                eng,
                pack,
                target,
                adversaries,
                existing,
                policy,
                deadline,
                &mut self.budget,
            )
        })();
        self.charge_compute(begin.elapsed());
        result
    }
    pub(in crate::margin_row) fn charge_compute(&mut self, elapsed: Duration) {
        self.elapsed = self.elapsed.saturating_add(elapsed);
    }
    pub(in crate::margin_row) fn receipt(&self) {
        let closed = self
            .prelude
            .as_ref()
            .map_or(0, |p| p.bounds.iter().filter(|b| **b > 0.0).count());
        let status = self.prelude.as_ref().map_or("declined", |p| {
            if p.complete() {
                "complete"
            } else {
                "partial"
            }
        });
        eprintln!("[direct-head] armed=1 status={status} closed={closed} calls={} native_passes={} native_columns={} elapsed_s={:.6}",
            self.calls, self.passes, self.columns, self.elapsed.as_secs_f64());
        self.adaptive.receipt();
        self.transport.receipt();
        if let Some(p) = &self.prelude {
            for (row, (&adversary, &lower)) in
                p.source.adversaries.iter().zip(&p.bounds).enumerate()
            {
                let leaves = self.adaptive.leaves.get(row).copied().unwrap_or(1);
                eprintln!("[direct-head-bound] adversary={adversary} cover_leaves={leaves} lower_bits={:016x}", lower.to_bits());
            }
        }
    }
}

fn scalar_slopes(q: &[I], y: &[I], deadline: Instant) -> Option<Vec<f64>> {
    q.iter()
        .zip(y)
        .map(|(&q, &y)| {
            if !live(deadline) {
                return None;
            }
            let mid = 0.5 * q.lo + 0.5 * q.hi;
            let alpha = if y.hi <= 0.0 {
                0.0
            } else if y.lo >= 0.0 {
                1.0
            } else if mid < 0.0 {
                y.hi / (y.hi - y.lo)
            } else {
                f64::from(u8::from(y.hi >= -y.lo))
            };
            let v = mid * alpha;
            good(v).then_some(v)
        })
        .collect()
}

// Search-only projection. No omitted feature, calibrated intercept or AY row
// can change the full-width scalar Y box or recipient's proof obligations.
fn export_lp(
    eng: &BackwardEngine<'_>,
    source: &DirectScalarHeadSource,
    q: &[I],
    deadline: Instant,
) -> Option<HeadFunctionalLpRequest> {
    let (w, _, (heads, width)) = eng.net.gemm1();
    let n = width.min(FUNCTIONAL_MAX_FEATURES);
    let selected: Vec<_> = (0..n)
        .map(|i| if n == 1 { 0 } else { i * (width - 1) / (n - 1) })
        .collect();
    let centers: Vec<_> = selected
        .iter()
        .map(|&k| 0.5 * source.features[k].lo + 0.5 * source.features[k].hi)
        .collect();
    let scale = width as f64 / n as f64;
    let boxes: Vec<_> = selected
        .iter()
        .zip(&centers)
        .map(|(&k, &c)| {
            let x = source.features[k];
            let r = (0.5 * x.hi - 0.5 * x.lo) * scale;
            [c - r, c + r]
        })
        .collect();
    let baseline = scalar_slopes(q, &source.y, deadline)?;
    let mut request = HeadFunctionalLpRequest {
        protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
        feature_count: n,
        bounds: boxes.clone(),
        objective: vec![0.0; n],
        rows: Vec::new(),
        slope_rows: vec![None; heads],
        phase_rows: Vec::new(),
    };
    for j in 0..heads {
        if !live(deadline) {
            return None;
        }
        let mut b = 0.5 * source.y[j].lo + 0.5 * source.y[j].hi;
        let mut row = Vec::with_capacity(n);
        for (i, &k) in selected.iter().enumerate() {
            if i & 63 == 0 && !live(deadline) {
                return None;
            }
            let a = w[j * width + k];
            b -= a * centers[i];
            row.push(a);
        }
        let nominal = 0.5 * q[j].lo + 0.5 * q[j].hi;
        if nominal > 0.0 {
            let v = request.bounds.len();
            let upper = row
                .iter()
                .zip(&boxes)
                .fold(b, |s, (&a, x)| s + a * if a >= 0.0 { x[1] } else { x[0] });
            if !good(upper) || !good(b) {
                return None;
            }
            request
                .bounds
                .push([0.0, upper.max(source.y[j].hi).max(0.0)]);
            request.objective.push(nominal);
            request.rows.push(HeadFunctionalLpRow {
                coefficients: vec![(v, 1.0)],
                lower: 0.0,
            });
            request.slope_rows[j] = Some(request.rows.len());
            let mut coefficients: Vec<_> = row
                .iter()
                .enumerate()
                .filter(|(_, a)| **a != 0.0)
                .map(|(k, &a)| (k, -a))
                .collect();
            coefficients.push((v, 1.0));
            request.rows.push(HeadFunctionalLpRow {
                coefficients,
                lower: b,
            });
        } else {
            for (k, &a) in row.iter().enumerate() {
                request.objective[k] += baseline[j] * a;
            }
        }
    }
    (live(deadline) && request.validate()).then_some(request)
}

fn ask<F>(
    eng: &BackwardEngine<'_>,
    source: &DirectScalarHeadSource,
    q: &[I],
    budget: &mut advice::AdviceBudget,
    calls: &mut usize,
    deadline: Instant,
    provider: F,
) -> Option<(HeadFunctionalLpRequest, HeadFunctionalLpHint)>
where
    F: FnOnce() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    let begin = Instant::now();
    let result = (|| {
        let call_deadline = budget.call_deadline(begin, deadline)?;
        let backend = std::panic::catch_unwind(std::panic::AssertUnwindSafe(provider))
            .ok()
            .flatten()?;
        let capable = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.supports_functional_advice()
        }))
        .unwrap_or(false);
        if !capable || !live(call_deadline) || !fp_qualified() {
            return None;
        }
        budget.calls += 1;
        let request = export_lp(eng, source, q, call_deadline)?;
        *calls += 1;
        let hint = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.propose_functional(&request, call_deadline)
        }))
        .ok()
        .flatten()?;
        (live(call_deadline) && fp_qualified() && hint.validate(&request))
            .then_some((request, hint))
    })();
    budget.charge_elapsed(begin.elapsed());
    result
}

struct DirectColumn {
    row: usize,
    slopes: Vec<f64>,
    offset: I,
}

fn column(
    source: &DirectScalarHeadSource,
    row: usize,
    q: &[I],
    bias: I,
    alpha: &[Option<f64>],
    deadline: Instant,
) -> Option<DirectColumn> {
    let mut slopes = scalar_slopes(q, &source.y, deadline)?;
    if alpha.len() != slopes.len() {
        return None;
    }
    let mut offset = bias;
    for j in 0..slopes.len() {
        if !live(deadline) {
            return None;
        }
        let mid = 0.5 * q[j].lo + 0.5 * q[j].hi;
        if mid > 0.0 {
            if let Some(a) = alpha[j] {
                slopes[j] = mid * a;
            }
        }
        offset = offset.add(I::p(scalar_intercept(q[j], slopes[j], source.y[j])?))?;
    }
    Some(DirectColumn {
        row,
        slopes,
        offset,
    })
}

fn score(
    eng: &BackwardEngine<'_>,
    source: &DirectScalarHeadSource,
    c: &DirectColumn,
    deadline: Instant,
) -> Option<f64> {
    let (w, b, (h, width)) = eng.net.gemm1();
    let n = width.min(FUNCTIONAL_MAX_FEATURES);
    let mut total = c.offset.lo;
    for j in 0..h {
        total += c.slopes[j] * b[j];
    }
    for i in 0..n {
        if !live(deadline) {
            return None;
        }
        let k = if n == 1 { 0 } else { i * (width - 1) / (n - 1) };
        let mut a = 0.0;
        for j in 0..h {
            a += c.slopes[j] * w[j * width + k];
        }
        let x = source.features[k];
        let center = 0.5 * x.lo + 0.5 * x.hi;
        let radius = (0.5 * x.hi - 0.5 * x.lo) * width as f64 / n as f64;
        total += a * center - a.abs() * radius;
    }
    (good(total) && live(deadline)).then_some(total)
}

pub(in crate::margin_row) fn attempt(
    eng: &BackwardEngine<'_>,
    target: usize,
    adversaries: &[usize],
    policy: HeadGroupPolicy,
    deadline: Instant,
) -> DirectAttempt {
    attempt_with_provider(
        eng,
        target,
        adversaries,
        policy,
        deadline,
        advice::registered_provider,
    )
}

fn attempt_with_provider<F>(
    eng: &BackwardEngine<'_>,
    target: usize,
    adversaries: &[usize],
    policy: HeadGroupPolicy,
    deadline: Instant,
    provider: F,
) -> DirectAttempt
where
    F: FnOnce() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    attempt_portfolio(eng, target, adversaries, policy, deadline, provider, true)
}

// The production portfolio always enables transport. The false branch is
// used only by tests of the inherited adaptive algorithm, so its split and
// incomplete-sibling regressions still exercise that actual implementation.
fn attempt_portfolio<F>(
    eng: &BackwardEngine<'_>,
    target: usize,
    adversaries: &[usize],
    policy: HeadGroupPolicy,
    deadline: Instant,
    provider: F,
    transport_enabled: bool,
) -> DirectAttempt
where
    F: FnOnce() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    let begin = Instant::now();
    let mut output = DirectAttempt {
        prelude: None,
        budget: advice::AdviceBudget::new(),
        calls: 0,
        columns: 0,
        passes: 0,
        elapsed: Duration::ZERO,
        adaptive: adaptive::Report::default(),
        transport: transport::Report::default(),
    };
    let result = (|| {
        let slice = begin
            .checked_add(Duration::from_secs_f64(policy.max_seconds()))?
            .min(deadline);
        let source = DirectScalarHeadSource::build(eng, target, adversaries, slice)?;
        let mut bounds = Vec::with_capacity(adversaries.len());
        let mut margins = Vec::with_capacity(adversaries.len());
        for row in 0..adversaries.len() {
            let (q, b) = source.margin(eng, row, slice)?;
            bounds.push(source.baseline(&q, b, slice)?);
            margins.push((q, b));
        }
        // A bounded native cover prepass shares this invocation's original
        // optional slice. It creates no provider allowance or YPack authority.
        // Large/expired contractions leave complete inherited floors intact.
        let native_begin = Instant::now();
        if let Some(remaining) = slice.checked_duration_since(native_begin) {
            let allowance = (remaining / 4).min(Duration::from_millis(250));
            if let Some(native_deadline) = native_begin.checked_add(allowance) {
                let native_deadline = native_deadline.min(slice);
                if transport_enabled {
                    // Pairing and phase covers share this one allowance;
                    // neither adds proposal calls or another wall-time slice.
                    output.transport = transport::refine(
                        eng,
                        &source,
                        &margins,
                        &mut bounds,
                        native_deadline,
                        deadline,
                    );
                }
                output.adaptive = adaptive::refine(
                    eng,
                    &source,
                    &margins,
                    &mut bounds,
                    policy,
                    native_deadline,
                    deadline,
                );
            }
        }
        // Provider construction is permitted only after this complete owned
        // unconditional cover, even when the numerical bounds remain open.
        let mut rows: Vec<_> = (0..bounds.len()).filter(|&r| bounds[r] <= 0.0).collect();
        rows.sort_by(|&a, &b| bounds[a].total_cmp(&bounds[b]).then(a.cmp(&b)));
        if let Some(&origin) = rows.first() {
            if let Some((request, hint)) = ask(
                eng,
                &source,
                &margins[origin].0,
                &mut output.budget,
                &mut output.calls,
                slice,
                provider,
            ) {
                let mut columns = Vec::with_capacity(MAX_DIRECT_NATIVE_COLUMNS);
                for &row in rows.iter().take(MAX_DIRECT_NATIVE_COLUMNS) {
                    let mut chosen = None;
                    let mut best = f64::NEG_INFINITY;
                    for sign in [1.0, -1.0] {
                        let alpha: Vec<_> = request
                            .slope_rows
                            .iter()
                            .enumerate()
                            .map(|(j, role)| {
                                let q =
                                    0.5 * margins[origin].0[j].lo + 0.5 * margins[origin].0[j].hi;
                                role.and_then(|r| {
                                    (q > 0.0)
                                        .then(|| (sign * hint.row_duals[r] / q).clamp(0.0, 1.0))
                                })
                            })
                            .collect();
                        if let Some(candidate) =
                            column(&source, row, &margins[row].0, margins[row].1, &alpha, slice)
                        {
                            if let Some(rank) = score(eng, &source, &candidate, slice) {
                                if rank > best {
                                    best = rank;
                                    chosen = Some(candidate);
                                }
                            }
                        }
                    }
                    if let Some(c) = chosen {
                        columns.push(c);
                    }
                }
                if !columns.is_empty() && live(slice) {
                    let mut seed = ndarray::Array2::zeros((source.y.len(), columns.len()));
                    for (r, c) in columns.iter().enumerate() {
                        for (j, &v) in c.slopes.iter().enumerate() {
                            seed[[j, r]] = v;
                        }
                    }
                    if let Ok(pass) =
                        eng.run_with_deadline(&Seed { s: seed, e: None }, LaneDir::Lower, slice)
                    {
                        output.passes = 1;
                        output.columns = columns.len();
                        for (r, c) in columns.iter().enumerate() {
                            if let Some(row) =
                                owned_row(&pass, &source.root, columns.len(), r, true, slice)
                            {
                                if let Some(lower) = row
                                    .lower(&source.root, slice)
                                    .and_then(|l| I::p(l).add(c.offset))
                                {
                                    bounds[c.row] = bounds[c.row].max(lower.lo);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Completed baseline/native bounds survive local expiry, but never
        // an expired original authority or a changed source generation.
        if !live(deadline)
            || !fp_qualified()
            || !source.seal.same_frame(eng, target, adversaries)
            || !live(deadline)
        {
            return None;
        }
        Some(DirectPrelude { source, bounds })
    })();
    output.prelude = result;
    output.elapsed = begin.elapsed();
    output
}

#[cfg(test)]
#[path = "direct_head_tests.rs"]
mod tests;
