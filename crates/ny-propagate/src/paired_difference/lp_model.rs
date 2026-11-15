// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Source-authenticated paired linear relaxation and solver-free dual replay.
//!
//! Shared input variables, exact authored affine equalities, rigorous forward
//! variable/strip bounds, independent ReLU triangles, and certified signed
//! paired supports form one immutable relaxation. An external NY proposal
//! optimizer may choose multipliers, but may not supply or replace any row.
//! This module checks a mathematical root band; CLI verdict authorization is
//! still owned by the existing formula/receipt gate.

use super::lp_certificate::{certified_lower, suitable_fp_environment, DualMultipliers, LinearRow};
use super::*;
use std::fmt::Write;

const MAX_VARIABLES: usize = 8_192;
const MAX_ROWS: usize = 100_000;
const MAX_TERMS: usize = 5_000_000;
const MAX_MAGNITUDE: f64 = 1e100;
const MAX_CERTIFICATE_BYTES: usize = 64 * 1024 * 1024;

#[cfg(test)]
std::thread_local! {
    // Per-thread evidence avoids interference between parallel unit tests.
    static SOURCE_MODEL_CONSTRUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The result of reconstructing source rows and checking both output signs.
/// `proved` means the complete supplied input box lies strictly inside the
/// band. It is mathematical evidence, not an authorized VNN-COMP receipt.
#[derive(Clone, Debug)]
pub struct LpProofReport {
    pub proved: bool,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub variables: usize,
    pub equalities: usize,
    pub inequalities: usize,
}

/// An immutable relaxation built ONLY from an admitted source chain/root box.
///
/// This type has no public constructor, mutable model access, or model-text
/// importer. Serialized rows are numeric advice for a proposal optimizer;
/// replay always reads this private, originally source-built model. Reports
/// describe precisely the source/domain captured at preparation, even if the
/// caller later mutates its chain or box. A different source requires a new
/// preparation. Neither the context nor its reports authorize a CLI verdict.
pub struct PreparedRootRelaxation {
    model: Model,
}

impl PreparedRootRelaxation {
    /// Serialize the captured source relaxation under the same live clock.
    /// This exports advice only and never imports replacement rows or boxes.
    pub fn dump_until(&self, deadline: Instant) -> Option<String> {
        if Instant::now() >= deadline {
            return None;
        }
        let text = dump_model(&self.model)?;
        (Instant::now() < deadline).then_some(text)
    }

    /// Check a bounded multiplier portfolio against the CAPTURED source model.
    /// No chain, box, row, objective, or claimed endpoint can be supplied here.
    pub fn check_texts_until(
        &self,
        epsilon: f64,
        texts: &[&str],
        deadline: Instant,
    ) -> Option<LpProofReport> {
        check_model_texts_until(&self.model, epsilon, texts, deadline)
    }
}

/// Source-recomputed bounds and dimensions of the stronger root relaxation.
/// This is mathematical evidence and never an authorized competition verdict.
#[derive(Clone, Debug)]
pub struct OutputTemplateCover {
    pub output: usize,
    pub sign: i8,
    pub portfolio: usize,
    pub upper: f64,
    pub nodes: usize,
    pub leaves: Vec<BoxDomain>,
}

#[derive(Clone, Debug)]
pub struct SoftGuardLeaf {
    pub domain: BoxDomain,
    pub guard: Option<usize>,
    pub beta: f64,
    pub root_upper: f64,
    pub guard_root_upper: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SoftGuardCover {
    pub output: usize,
    pub sign: i8,
    pub portfolio: usize,
    pub upper: f64,
    pub nodes: usize,
    pub leaves: Vec<SoftGuardLeaf>,
}

#[derive(Clone, Debug)]
pub struct RootRelaxationReport {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub variables: usize,
    pub equalities: usize,
    pub baseline_inequalities: usize,
    pub inequalities: usize,
    pub cover_cells: usize,
    pub propagation_rows: usize,
    pub affine_hull_rows: usize,
    pub group_rows: usize,
    pub coupled_affine_rows: usize,
    pub cover_rows: usize,
    pub template_rows: usize,
    pub output_template_rows: usize,
    pub output_template_covers: Vec<OutputTemplateCover>,
    pub soft_guard_rows: usize,
    pub soft_guard_covers: Vec<SoftGuardCover>,
}

#[derive(Clone)]
pub(super) struct Model {
    pub(super) bounds: Vec<(f64, f64)>,
    pub(super) equalities: Vec<LinearRow>,
    pub(super) pivots: Vec<usize>,
    pub(super) inequalities: Vec<LinearRow>,
    pub(super) outputs: Vec<(usize, usize)>,
    pub(super) output_lo: Vec<f64>,
    pub(super) output_hi: Vec<f64>,
    // Source stage and neuron bindings, never read from a proposal artifact.
    // Tuple order: pre_f, pre_g, post_f, post_g.
    pub(super) relu_pairs: Vec<(usize, Vec<(usize, usize, usize, usize)>)>,
    pub(super) baseline_inequalities: usize,
    pub(super) cover_cells: usize,
    pub(super) propagation_rows: usize,
    pub(super) affine_hull_rows: usize,
    pub(super) group_rows: usize,
    pub(super) coupled_affine_rows: usize,
    pub(super) cover_rows: usize,
    pub(super) template_rows: usize,
    pub(super) output_template_rows: usize,
    pub(super) output_template_covers: Vec<OutputTemplateCover>,
    pub(super) soft_guard_rows: usize,
    pub(super) soft_guard_covers: Vec<SoftGuardCover>,
    terms: usize,
}

fn admissible(x: f64) -> bool {
    x.is_finite() && x.abs() <= MAX_MAGNITUDE
}

impl Model {
    fn variables(&mut self, lo: &[f64], hi: &[f64]) -> Option<Vec<usize>> {
        if lo.len() != hi.len() || self.bounds.len().checked_add(lo.len())? > MAX_VARIABLES {
            return None;
        }
        let first = self.bounds.len();
        for (&l, &h) in lo.iter().zip(hi) {
            if !admissible(l) || !admissible(h) || l > h {
                return None;
            }
            self.bounds.push((l, h));
        }
        Some((first..self.bounds.len()).collect())
    }

    pub(super) fn row(
        &mut self,
        terms: Vec<(usize, f64)>,
        rhs: f64,
        pivot: Option<usize>,
    ) -> Option<()> {
        let terms: Vec<_> = terms.into_iter().filter(|&(_, a)| a != 0.0).collect();
        if !admissible(rhs)
            || self.equalities.len().checked_add(self.inequalities.len())? >= MAX_ROWS
            || terms
                .iter()
                .any(|&(j, a)| j >= self.bounds.len() || !admissible(a))
        {
            return None;
        }
        self.terms = self.terms.checked_add(terms.len())?;
        if self.terms > MAX_TERMS {
            return None;
        }
        let row = LinearRow { terms, rhs };
        if let Some(p) = pivot {
            // The optimizer can eliminate these authored target equations in
            // reverse order. Their exact algebra remains checked in full.
            if row.terms.iter().filter(|&&(j, _)| j == p).count() != 1
                || !row.terms.iter().any(|&(j, a)| j == p && a == 1.0)
                || row.terms.iter().any(|&(j, _)| j > p)
                || self.pivots.contains(&p)
            {
                return None;
            }
            self.pivots.push(p);
            self.equalities.push(row);
        } else {
            self.inequalities.push(row);
        }
        Some(())
    }

    fn relu(&mut self, source: usize, target: usize, l: f64, u: f64) -> Option<()> {
        if u <= 0.0 {
            return self.row(vec![(target, 1.0)], 0.0, Some(target));
        }
        if l >= 0.0 {
            return self.row(vec![(target, 1.0), (source, -1.0)], 0.0, Some(target));
        }
        self.row(vec![(target, 1.0)], 0.0, None)?;
        self.row(vec![(target, 1.0), (source, -1.0)], 0.0, None)?;
        let slope = u / (u - l);
        if !admissible(slope) {
            return None;
        }
        // A rounded secant is merely a slope proposal. The exact f64 row's
        // intercept is repaired over all pieces of the source ReLU graph.
        let k = independent_intercept(-1.0, -slope, l, u, 0.0, 0.0, 0.0, 0.0);
        self.row(vec![(source, slope), (target, -1.0)], k, None)
    }

    fn strip(&mut self, f: &[usize], g: &[usize], state: &Enclosure) -> Option<()> {
        if f.len() != g.len() || f.len() != state.d_lo.len() || f.len() != state.d_hi.len() {
            return None;
        }
        for j in 0..f.len() {
            self.row(vec![(f[j], 1.0), (g[j], -1.0)], state.d_lo[j], None)?;
            self.row(vec![(g[j], 1.0), (f[j], -1.0)], -state.d_hi[j], None)?;
        }
        Some(())
    }

    fn meet_bounds(&mut self, indices: &[usize], lo: &[f64], hi: &[f64]) -> Option<()> {
        if indices.len() != lo.len() || lo.len() != hi.len() {
            return None;
        }
        for ((&index, &l), &u) in indices.iter().zip(lo).zip(hi) {
            let bound = self.bounds.get_mut(index)?;
            if !admissible(l) || !admissible(u) || l > u {
                return None;
            }
            bound.0 = bound.0.max(l);
            bound.1 = bound.1.min(u);
            if bound.0 > bound.1 {
                return None;
            }
        }
        Some(())
    }

    /// Append a tighter scalar hull without replacing any baseline row or
    /// authored pivot. A newly stable unit must not change variable allocation.
    fn strengthen_relu(&mut self, source: usize, target: usize, l: f64, u: f64) -> Option<()> {
        if u <= 0.0 {
            return self.row(vec![(target, -1.0)], 0.0, None);
        }
        if l >= 0.0 {
            return self.row(vec![(source, 1.0), (target, -1.0)], 0.0, None);
        }
        let slope = u / (u - l);
        if !admissible(slope) {
            return None;
        }
        let k = independent_intercept(-1.0, -slope, l, u, 0.0, 0.0, 0.0, 0.0);
        self.row(vec![(source, slope), (target, -1.0)], k, None)
    }
}

pub(super) fn build(chain: &PairedChain, domain: &BoxDomain) -> Option<Model> {
    build_until(chain, domain, None)
}

pub(super) fn build_until(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
) -> Option<Model> {
    let model = build_with_cuts_until(chain, domain, deadline, true)?;
    #[cfg(test)]
    SOURCE_MODEL_CONSTRUCTIONS.with(|count| count.set(count.get() + 1));
    Some(model)
}

fn build_with_cuts_until(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    strong: bool,
) -> Option<Model> {
    if !suitable_fp_environment() {
        return None;
    }
    chain.validate().ok()?;
    // The baseline fixes every column, equality and inequality. New bounds and
    // planes are intersections with this model, never replacements for it.
    let states = refine::states(chain, domain, deadline, true)?;
    if states.len() != chain.stages.len() + 1 {
        return None;
    }
    let output = states.last()?;
    let mut model = Model {
        bounds: Vec::new(),
        equalities: Vec::new(),
        pivots: Vec::new(),
        inequalities: Vec::new(),
        outputs: Vec::new(),
        output_lo: output.d_lo.clone(),
        output_hi: output.d_hi.clone(),
        relu_pairs: Vec::new(),
        baseline_inequalities: 0,
        cover_cells: 0,
        propagation_rows: 0,
        affine_hull_rows: 0,
        group_rows: 0,
        coupled_affine_rows: 0,
        cover_rows: 0,
        template_rows: 0,
        output_template_rows: 0,
        output_template_covers: Vec::new(),
        soft_guard_rows: 0,
        soft_guard_covers: Vec::new(),
        terms: 0,
    };
    // Both towers reference the very same root variable indices.
    let mut f = model.variables(&domain.lo, &domain.hi)?;
    let mut g = f.clone();
    let mut bindings = vec![(f.clone(), g.clone())];
    let mut delta_bindings = vec![vec![None; f.len()]];
    for (s, stage) in chain.stages.iter().enumerate() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        let source = &states[s];
        let target = &states[s + 1];
        let nf = model.variables(&target.f_lo, &target.f_hi)?;
        // Use (common f, signed delta) as the numerical free basis whenever
        // g's ReLU is unstable. The auxiliary variable denotes EXACTLY f-g;
        // its already certified signed box preserves the paired correlation
        // directly in a box-residual dual. Allocate it before g so g=f-d has
        // an authored +1 target pivot with strictly earlier source columns.
        let mut delta = vec![None; nf.len()];
        if let Stage::Relu { width } = stage {
            let unstable: Vec<_> = (0..*width)
                .filter(|&j| source.g_lo[j] < 0.0 && source.g_hi[j] > 0.0)
                .collect();
            let lo: Vec<_> = unstable.iter().map(|&j| target.d_lo[j]).collect();
            let hi: Vec<_> = unstable.iter().map(|&j| target.d_hi[j]).collect();
            let indices = model.variables(&lo, &hi)?;
            for (&j, index) in unstable.iter().zip(indices) {
                delta[j] = Some(index);
            }
        }
        let ng = model.variables(&target.g_lo, &target.g_hi)?;
        match stage {
            Stage::Affine {
                rows,
                cols,
                wf,
                wg,
                bf,
                bg,
            } => {
                for (weights, bias, inputs, outputs) in [(wf, bf, &f, &nf), (wg, bg, &g, &ng)] {
                    for i in 0..*rows {
                        if deadline.is_some_and(|d| Instant::now() >= d) {
                            return None;
                        }
                        let mut terms = Vec::with_capacity(cols + 1);
                        terms.push((outputs[i], 1.0));
                        for j in 0..*cols {
                            terms.push((inputs[j], -weights[i * cols + j]));
                        }
                        // Coefficients/bias denote the exact authored f64
                        // values; no affine composition or coefficient sum.
                        model.row(terms, bias[i], Some(outputs[i]))?;
                    }
                }
            }
            Stage::Relu { width } => {
                model
                    .relu_pairs
                    .push((s, (0..*width).map(|j| (f[j], g[j], nf[j], ng[j])).collect()));
                for j in 0..*width {
                    model.relu(f[j], nf[j], source.f_lo[j], source.f_hi[j])?;
                }
                for j in 0..*width {
                    model.relu(g[j], ng[j], source.g_lo[j], source.g_hi[j])?;
                }
                for j in 0..*width {
                    if let Some(d) = delta[j] {
                        // Every actual tower evaluation extends to this
                        // auxiliary variable by d=f-g. This is exact algebra,
                        // with no authored weight composition or rounding.
                        model.row(
                            vec![(ng[j], 1.0), (nf[j], -1.0), (d, 1.0)],
                            0.0,
                            Some(ng[j]),
                        )?;
                    }
                }
                for j in 0..*width {
                    if deadline.is_some_and(|d| Instant::now() >= d) {
                        return None;
                    }
                    let ranges = PairedReluRanges {
                        lf: source.f_lo[j],
                        uf: source.f_hi[j],
                        lg: source.g_lo[j],
                        ug: source.g_hi[j],
                        dl: source.d_lo[j],
                        du: source.d_hi[j],
                    };
                    // Stable paired activations already have exact authored
                    // equations and source strips; no joint planes needed.
                    if (ranges.lf >= 0.0 || ranges.uf <= 0.0)
                        && (ranges.lg >= 0.0 || ranges.ug <= 0.0)
                    {
                        continue;
                    }
                    let hull = source.hulls.as_ref()?.get(j)?;
                    for (cf, cg) in [(1.0, -1.0), (-1.0, 1.0)] {
                        for (af, ag) in ranges.candidate_slopes(cf, cg) {
                            // Arbitrary rounded slopes gain authority ONLY
                            // from this prepared vertex replay.
                            let k = hull.certified_intercept(cf, cg, af, ag)?;
                            model.row(
                                vec![(nf[j], cf), (ng[j], cg), (f[j], -af), (g[j], -ag)],
                                k,
                                None,
                            )?;
                        }
                    }
                }
            }
        }
        // Certified signed intermediate strips are explicit coupling rows.
        model.strip(&nf, &ng, target)?;
        f = nf;
        g = ng;
        bindings.push((f.clone(), g.clone()));
        delta_bindings.push(delta);
    }
    if f.len() != chain.output_dim || g.len() != chain.output_dim {
        return None;
    }
    model.outputs = f.into_iter().zip(g).collect();
    model.baseline_inequalities = model.inequalities.len();
    if strong {
        let states = refine::states_strong_from_reference(chain, domain, deadline, true, &states)?;
        for s in 0..states.len() {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            let target = &states[s];
            let (f, g) = &bindings[s];
            model.meet_bounds(f, &target.f_lo, &target.f_hi)?;
            model.meet_bounds(g, &target.g_lo, &target.g_hi)?;
            for (j, index) in delta_bindings[s].iter().enumerate() {
                if let Some(index) = index {
                    model.meet_bounds(&[*index], &[target.d_lo[j]], &[target.d_hi[j]])?;
                }
            }
            if s > 0 {
                model.strip(f, g, target)?;
            }
        }
        let output = states.last()?;
        for j in 0..chain.output_dim {
            model.output_lo[j] = model.output_lo[j].max(output.d_lo[j]);
            model.output_hi[j] = model.output_hi[j].min(output.d_hi[j]);
        }
        model.propagation_rows = model.inequalities.len() - model.baseline_inequalities;
        // New unconditional rows append to the unchanged baseline relaxation.
        // Source variables/equalities retain their IDs and authored semantics.
        for (s, stage) in chain.stages.iter().enumerate() {
            if let Stage::Relu { width } = stage {
                let source = &states[s];
                let (pre_f, pre_g) = &bindings[s];
                let (post_f, post_g) = &bindings[s + 1];
                let before = model.inequalities.len();
                for j in 0..*width {
                    model.strengthen_relu(pre_f[j], post_f[j], source.f_lo[j], source.f_hi[j])?;
                    model.strengthen_relu(pre_g[j], post_g[j], source.g_lo[j], source.g_hi[j])?;
                }
                model.propagation_rows += model.inequalities.len() - before;
                let affine_rows = lp_affine_hull::rows(chain, &states, s, &bindings, deadline)?;
                model.affine_hull_rows += affine_rows.len();
                for row in affine_rows {
                    model.row(row.terms, row.rhs, None)?;
                }
                let group_rows = lp_group::rows(
                    chain, domain, &states, s, pre_f, pre_g, post_f, post_g, deadline,
                )?;
                model.group_rows += group_rows.len();
                for row in group_rows {
                    model.row(row.terms, row.rhs, None)?;
                }
                let coupled_rows =
                    lp_coupled_affine::rows(chain, domain, &states, s, &bindings, deadline)?;
                model.coupled_affine_rows += coupled_rows.len();
                for row in coupled_rows {
                    model.row(row.terms, row.rhs, None)?;
                }
            }
        }
        let cover = lp_cover::rows(chain, domain, &states, &bindings, deadline)?;
        model.cover_cells = cover.cells;
        model.cover_rows = cover.rows.len();
        model.output_lo = cover.output_lower;
        model.output_hi = cover.output_upper;
        for row in cover.rows {
            model.row(row.terms, row.rhs, None)?;
        }
        // Whole-prefix source identities couple this activation to every
        // earlier ReLU. Arbitrary source-derived masks have no phase premise:
        // their nonnegative residual slacks make every lifted row global.
        let (template_rows, output_rows, guard_rows, output_covers, guard_covers) =
            lp_template::rows_with_output_and_guards(chain, domain, &bindings, deadline)?;
        model.template_rows = template_rows.len();
        for row in template_rows {
            model.row(row.terms, row.rhs, None)?;
        }
        // Aggregate signed output terms before bounding the complete root
        // cover. The same source-authenticated nonnegative causal slacks
        // repair every earlier template mismatch; no activation premise is
        // introduced. Append after every original template row so the old
        // relaxation remains an exact prefix of this stronger model.
        model.output_template_rows = output_rows.len();
        for row in output_rows {
            model.row(row.terms, row.rhs, None)?;
        }
        model.output_template_covers = output_covers
            .into_iter()
            .map(|cover| OutputTemplateCover {
                output: cover.output,
                sign: cover.sign,
                portfolio: cover.portfolio,
                upper: cover.upper,
                nodes: cover.nodes,
                leaves: cover.leaves,
            })
            .collect();
        // Charged covers add larger mismatch penalties and tighter root
        // intercepts. Append beside the original rows to retain dominance.
        model.soft_guard_rows = guard_rows.len();
        for row in guard_rows {
            model.row(row.terms, row.rhs, None)?;
        }
        model.soft_guard_covers = guard_covers
            .into_iter()
            .map(|cover| SoftGuardCover {
                output: cover.output,
                sign: cover.sign,
                portfolio: cover.portfolio,
                upper: cover.upper,
                nodes: cover.nodes,
                leaves: cover
                    .leaves
                    .into_iter()
                    .map(|leaf| SoftGuardLeaf {
                        domain: leaf.domain,
                        guard: leaf.guard,
                        beta: leaf.beta,
                        root_upper: leaf.root_upper,
                        guard_root_upper: leaf.guard_root_upper,
                    })
                    .collect(),
            })
            .collect();
    }
    deadline.is_none_or(|d| Instant::now() < d).then_some(model)
}

/// Replays proposals against the single immutable SOURCE-CONSTRUCTED model.
/// Sibling native optimizers supply numbers only, in output (+,-) order.
pub(super) fn check_candidates(
    model: &Model,
    epsilon: f64,
    candidates: &[DualMultipliers],
    deadline: Option<Instant>,
) -> Option<LpProofReport> {
    if !admissible(epsilon) || epsilon <= 0.0 || candidates.len() != 2 * model.outputs.len() {
        return None;
    }
    let mut lower = model.output_lo.clone();
    let mut upper = model.output_hi.clone();
    let mut c = vec![0.0; model.bounds.len()];
    for (j, &(f, g)) in model.outputs.iter().enumerate() {
        for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            c[f] = sign;
            c[g] = -sign;
            let value = certified_lower(
                &model.bounds,
                &c,
                &model.equalities,
                &model.inequalities,
                &candidates[2 * j + side],
            )?;
            if sign > 0.0 {
                lower[j] = lower[j].max(value);
            } else {
                upper[j] = upper[j].min(-value);
            }
            c[f] = 0.0;
            c[g] = 0.0;
        }
        if !lower[j].is_finite() || !upper[j].is_finite() || lower[j] > upper[j] {
            return None;
        }
    }
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return None;
    }
    let proved = lower
        .iter()
        .zip(&upper)
        .all(|(&l, &h)| l > -epsilon && h < epsilon);
    Some(LpProofReport {
        proved,
        lower,
        upper,
        variables: model.bounds.len(),
        equalities: model.equalities.len(),
        inequalities: model.inequalities.len(),
    })
}

fn dump_row(text: &mut String, row: &LinearRow, pivot: Option<usize>) -> Option<()> {
    if let Some(pivot) = pivot {
        write!(text, "{pivot} ").ok()?;
    }
    write!(text, "{}", row.terms.len()).ok()?;
    for &(j, a) in &row.terms {
        write!(text, " {j} {a:.17e}").ok()?;
    }
    writeln!(text, " {:.17e}", row.rhs).ok()?;
    Some(())
}

fn number(tokens: &mut std::str::SplitWhitespace<'_>) -> Option<f64> {
    let value = tokens.next()?.parse::<f64>().ok()?;
    admissible(value).then_some(value)
}

fn multipliers(
    tokens: &mut std::str::SplitWhitespace<'_>,
    eq: usize,
    ineq: usize,
) -> Option<DualMultipliers> {
    let equality_multipliers = (0..eq)
        .map(|_| number(tokens))
        .collect::<Option<Vec<_>>>()?;
    let inequality_multipliers = (0..ineq)
        .map(|_| number(tokens))
        .collect::<Option<Vec<_>>>()?;
    if inequality_multipliers.iter().any(|&x| x < 0.0) {
        return None;
    }
    Some(DualMultipliers {
        equality_multipliers,
        inequality_multipliers,
    })
}

/// Serialize only this source-owned immutable relaxation.
pub(super) fn dump_model(model: &Model) -> Option<String> {
    let mut text = String::new();
    writeln!(
        text,
        "PAIRED_LP_GEQ_V1 {} {} {} {}",
        model.bounds.len(),
        model.equalities.len(),
        model.inequalities.len(),
        model.outputs.len()
    )
    .ok()?;
    for &(l, h) in &model.bounds {
        writeln!(text, "{l:.17e} {h:.17e}").ok()?;
    }
    for (row, &pivot) in model.equalities.iter().zip(&model.pivots) {
        dump_row(&mut text, row, Some(pivot))?;
    }
    for row in &model.inequalities {
        dump_row(&mut text, row, None)?;
    }
    for &(f, g) in &model.outputs {
        writeln!(text, "{f} {g}").ok()?;
    }
    Some(text)
}

/// Candidate files contain numbers only: row/bound/objective ownership stays
/// with the rebuilt source model, including all conditional phase rows.
pub(super) fn parse_dual_text(model: &Model, text: &str) -> Option<Vec<DualMultipliers>> {
    if text.len() > MAX_CERTIFICATE_BYTES {
        return None;
    }
    let mut tokens = text.split_whitespace();
    if tokens.next()? != "PAIRED_LP_DUAL_GEQ_V1"
        || tokens.next()?.parse::<usize>().ok()? != model.equalities.len()
        || tokens.next()?.parse::<usize>().ok()? != model.inequalities.len()
        || tokens.next()?.parse::<usize>().ok()? != model.outputs.len()
    {
        return None;
    }
    let candidates = (0..2 * model.outputs.len())
        .map(|_| {
            multipliers(
                &mut tokens,
                model.equalities.len(),
                model.inequalities.len(),
            )
        })
        .collect::<Option<Vec<_>>>()?;
    if tokens.next().is_some() {
        return None;
    }
    Some(candidates)
}

fn check_model_texts_until(
    model: &Model,
    epsilon: f64,
    texts: &[&str],
    deadline: Instant,
) -> Option<LpProofReport> {
    let bytes = texts
        .iter()
        .try_fold(0usize, |n, text| n.checked_add(text.len()))?;
    if !admissible(epsilon)
        || epsilon <= 0.0
        || texts.is_empty()
        || texts.len() > 4
        || bytes > MAX_CERTIFICATE_BYTES
        || Instant::now() >= deadline
    {
        return None;
    }
    let mut best: Option<LpProofReport> = None;
    for &text in texts {
        let candidates = parse_dual_text(model, text)?;
        let report = check_candidates(model, epsilon, &candidates, Some(deadline))?;
        if let Some(best) = best.as_mut() {
            for j in 0..model.outputs.len() {
                best.lower[j] = best.lower[j].max(report.lower[j]);
                best.upper[j] = best.upper[j].min(report.upper[j]);
                if best.lower[j] > best.upper[j] {
                    return None;
                }
            }
        } else {
            best = Some(report);
        }
    }
    let mut best = best?;
    best.proved = best
        .lower
        .iter()
        .zip(&best.upper)
        .all(|(&l, &h)| l > -epsilon && h < epsilon);
    (Instant::now() < deadline).then_some(best)
}

impl PairedChain {
    /// Prepare one opaque, immutable source/root relaxation for proposal and
    /// replay. Every authority-bearing field is built here from `self` and
    /// `domain`; public callers cannot import a previously serialized model.
    pub fn prepare_root_lp_until(
        &self,
        domain: &BoxDomain,
        deadline: Instant,
    ) -> Option<PreparedRootRelaxation> {
        let model = build_until(self, domain, Some(deadline))?;
        (Instant::now() < deadline).then_some(PreparedRootRelaxation { model })
    }

    /// Frozen old relaxation for controlled source-identical comparisons.
    /// It omits the new global/inter-neuron cuts and cannot certify their bounds.
    pub fn lp_dump_baseline_until(&self, domain: &BoxDomain, deadline: Instant) -> Option<String> {
        let text = dump_model(&build_with_cuts_until(self, domain, Some(deadline), false)?)?;
        (Instant::now() < deadline).then_some(text)
    }

    /// Build and inspect the source-certified stronger root relaxation.
    pub fn root_relaxation_until(
        &self,
        domain: &BoxDomain,
        deadline: Instant,
    ) -> Option<RootRelaxationReport> {
        let model = build_until(self, domain, Some(deadline))?;
        Some(RootRelaxationReport {
            lower: model.output_lo,
            upper: model.output_hi,
            variables: model.bounds.len(),
            equalities: model.equalities.len(),
            baseline_inequalities: model.baseline_inequalities,
            inequalities: model.inequalities.len(),
            cover_cells: model.cover_cells,
            propagation_rows: model.propagation_rows,
            affine_hull_rows: model.affine_hull_rows,
            group_rows: model.group_rows,
            coupled_affine_rows: model.coupled_affine_rows,
            cover_rows: model.cover_rows,
            template_rows: model.template_rows,
            output_template_rows: model.output_template_rows,
            output_template_covers: model.output_template_covers,
            soft_guard_rows: model.soft_guard_rows,
            soft_guard_covers: model.soft_guard_covers,
        })
    }

    /// Export an authenticated relaxation for an NY/AY proposal optimizer.
    /// Bounds, authored equalities, GEQ rows and output pairs retain source order.
    pub fn lp_dump(&self, domain: &BoxDomain) -> Option<String> {
        dump_model(&build(self, domain)?)
    }

    /// Timed source reconstruction. Expiry always declines; this API does not
    /// promise interruption during any single arithmetic/serialization loop.
    pub fn lp_dump_until(&self, domain: &BoxDomain, deadline: Instant) -> Option<String> {
        self.prepare_root_lp_until(domain, deadline)?
            .dump_until(deadline)
    }

    /// Reconstruct source rows and check untrusted multipliers. Format:
    /// PAIRED_LP_DUAL_GEQ_V1 ne ni nout; then per output (+,-), ne free
    /// equality multipliers followed by ni nonnegative GEQ multipliers.
    pub fn check_lp_text(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        text: &str,
    ) -> Option<LpProofReport> {
        if !admissible(epsilon) || epsilon <= 0.0 || text.len() > MAX_CERTIFICATE_BYTES {
            return None;
        }
        let model = build(self, domain)?;
        let candidates = parse_dual_text(&model, text)?;
        check_candidates(&model, epsilon, &candidates, None)
    }

    /// Live-verdict callers share their original absolute deadline with source
    /// reconstruction and replay. No late candidate can mint a timely proof.
    pub fn check_lp_text_until(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        text: &str,
        deadline: Instant,
    ) -> Option<LpProofReport> {
        self.check_lp_texts_until(domain, epsilon, &[text], deadline)
    }

    /// Rebuild once and meet a bounded portfolio of independent proposals.
    /// Every endpoint is re-derived against this same source/root/objective;
    /// callers cannot introduce a claimed endpoint into the intersection.
    pub fn check_lp_texts_until(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        texts: &[&str],
        deadline: Instant,
    ) -> Option<LpProofReport> {
        let bytes = texts
            .iter()
            .try_fold(0usize, |n, text| n.checked_add(text.len()))?;
        if !admissible(epsilon)
            || epsilon <= 0.0
            || texts.is_empty()
            || texts.len() > 4
            || bytes > MAX_CERTIFICATE_BYTES
            || Instant::now() >= deadline
        {
            return None;
        }
        self.prepare_root_lp_until(domain, deadline)?
            .check_texts_until(epsilon, texts, deadline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> PairedChain {
        PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![1.0],
                    wg: vec![1.0],
                    bf: vec![0.0],
                    bg: vec![0.25],
                },
                Stage::Relu { width: 1 },
            ],
        }
    }
    fn domain() -> BoxDomain {
        BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        }
    }
    fn zero_certificate(model: &Model) -> String {
        let mut text = format!(
            "PAIRED_LP_DUAL_GEQ_V1 {} {} {}\n",
            model.equalities.len(),
            model.inequalities.len(),
            model.outputs.len()
        );
        for _ in 0..2 * model.outputs.len() * (model.equalities.len() + model.inequalities.len()) {
            text.push_str("0 ");
        }
        text
    }

    fn prepared_affine_fixture() -> PairedChain {
        PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![1.125],
                wg: vec![1.0],
                bf: vec![0.25],
                bg: vec![0.0],
            }],
        }
    }

    #[test]
    fn prepared_root_replay_reuses_one_source_model_and_matches_reconstruction() {
        let chain = chain();
        let domain = domain();
        let deadline = Instant::now() + Duration::from_secs(20);
        let before = SOURCE_MODEL_CONSTRUCTIONS.with(std::cell::Cell::get);
        let prepared = chain.prepare_root_lp_until(&domain, deadline).unwrap();
        let after = SOURCE_MODEL_CONSTRUCTIONS.with(std::cell::Cell::get);
        assert_eq!(after, before + 1);
        let dump = prepared.dump_until(deadline).unwrap();
        let text = zero_certificate(&prepared.model);
        let cached = prepared
            .check_texts_until(0.5, &[&text, &text], deadline)
            .unwrap();
        assert!(cached.proved);
        assert_eq!(prepared.dump_until(deadline).unwrap(), dump);
        assert_eq!(SOURCE_MODEL_CONSTRUCTIONS.with(std::cell::Cell::get), after);
        let reconstructed = chain
            .check_lp_texts_until(&domain, 0.5, &[&text, &text], deadline)
            .unwrap();
        assert_eq!(
            SOURCE_MODEL_CONSTRUCTIONS.with(std::cell::Cell::get),
            after + 1
        );
        assert_eq!(cached.proved, reconstructed.proved);
        assert_eq!(
            cached.lower.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            reconstructed
                .lower
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            cached.upper.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            reconstructed
                .upper
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            (cached.variables, cached.equalities, cached.inequalities),
            (
                reconstructed.variables,
                reconstructed.equalities,
                reconstructed.inequalities
            )
        );
        assert_eq!(dump, chain.lp_dump_until(&domain, deadline).unwrap());
    }

    #[test]
    fn prepared_root_reports_only_its_captured_source_and_domain_after_mutation() {
        let mut chain = prepared_affine_fixture();
        let mut domain = domain();
        let deadline = Instant::now() + Duration::from_secs(20);
        let prepared = chain.prepare_root_lp_until(&domain, deadline).unwrap();
        let original = prepared.dump_until(deadline).unwrap();
        let old_text = zero_certificate(&prepared.model);
        assert!(
            prepared
                .check_texts_until(0.4, &[&old_text], deadline)
                .unwrap()
                .proved
        );
        // A real source violation of the tighter band must stay unproved.
        let (f, g) = chain.forward(&[1.0]).unwrap();
        assert_eq!(f[0] - g[0], 0.375);
        assert!(
            !prepared
                .check_texts_until(0.3, &[&old_text], deadline)
                .unwrap()
                .proved
        );

        let Stage::Affine { bf, .. } = &mut chain.stages[0] else {
            unreachable!()
        };
        bf[0] = 0.5;
        let changed = chain.prepare_root_lp_until(&domain, deadline).unwrap();
        let changed_text = zero_certificate(&changed.model);
        let (f, g) = chain.forward(&[1.0]).unwrap();
        assert_eq!(f[0] - g[0], 0.625);
        assert!(
            !changed
                .check_texts_until(0.4, &[&changed_text], deadline)
                .unwrap()
                .proved
        );

        chain = prepared_affine_fixture();
        domain.lo[0] = -2.0;
        domain.hi[0] = 2.0;
        let widened = chain.prepare_root_lp_until(&domain, deadline).unwrap();
        let widened_text = zero_certificate(&widened.model);
        let (f, g) = chain.forward(&[2.0]).unwrap();
        assert_eq!(f[0] - g[0], 0.5);
        assert!(
            !widened
                .check_texts_until(0.4, &[&widened_text], deadline)
                .unwrap()
                .proved
        );
        // The old context cannot accept a new source or domain argument.
        // It remains mathematical evidence for its originally captured pair.
        assert_eq!(prepared.dump_until(deadline).unwrap(), original);
        assert!(
            prepared
                .check_texts_until(0.4, &[&old_text], deadline)
                .unwrap()
                .proved
        );
    }

    #[test]
    fn prepared_root_rejects_foreign_rows_and_malformed_multiplier_authority() {
        let chain = chain();
        let deadline = Instant::now() + Duration::from_secs(20);
        let prepared = chain.prepare_root_lp_until(&domain(), deadline).unwrap();
        let text = zero_certificate(&prepared.model);
        let model_text = prepared.dump_until(deadline).unwrap();
        // A model transcript is NOT a multiplier certificate or an importer.
        assert!(prepared
            .check_texts_until(0.5, &[&model_text], deadline)
            .is_none());
        let appended = text.clone() + " replacement_row replacement_bound";
        assert!(prepared
            .check_texts_until(0.5, &[&appended], deadline)
            .is_none());
        let mut tokens: Vec<_> = text.split_whitespace().map(str::to_owned).collect();
        tokens[4 + prepared.model.equalities.len()] = "-1".into();
        let negative_geq = tokens.join(" ");
        assert!(prepared
            .check_texts_until(0.5, &[&negative_geq], deadline)
            .is_none());
        let nonfinite = text.replace("0 ", "NaN ");
        assert!(prepared
            .check_texts_until(0.5, &[&nonfinite], deadline)
            .is_none());
        assert!(prepared
            .check_texts_until(0.5, &[&text, &appended], deadline)
            .is_none());
        assert!(
            prepared
                .check_texts_until(0.5, &[&text], deadline)
                .unwrap()
                .proved
        );
    }

    #[test]
    fn prepared_root_retains_strict_epsilon_and_absolute_expiry_guards() {
        let chain = prepared_affine_fixture();
        let domain = domain();
        let deadline = Instant::now() + Duration::from_secs(20);
        let prepared = chain.prepare_root_lp_until(&domain, deadline).unwrap();
        let text = zero_certificate(&prepared.model);
        assert!(
            !prepared
                .check_texts_until(0.375, &[&text], deadline)
                .unwrap()
                .proved
        );
        assert!(
            prepared
                .check_texts_until(0.4, &[&text], deadline)
                .unwrap()
                .proved
        );
        for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(prepared
                .check_texts_until(epsilon, &[&text], deadline)
                .is_none());
        }
        assert!(prepared.check_texts_until(0.4, &[], deadline).is_none());
        assert!(prepared
            .check_texts_until(0.4, &[text.as_str(); 5], deadline)
            .is_none());
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        assert!(chain.prepare_root_lp_until(&domain, expired).is_none());
        assert!(prepared.dump_until(expired).is_none());
        assert!(prepared.check_texts_until(0.4, &[&text], expired).is_none());
    }

    #[test]
    fn stronger_model_retains_every_baseline_row_and_column_when_g_becomes_stable() {
        let chain = PairedChain {
            input_dim: 2,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 2,
                    cols: 2,
                    wf: vec![0.0; 4],
                    wg: vec![1.0, 1.0, 0.0, 1.0],
                    bf: vec![0.0; 2],
                    bg: vec![-1.0, 0.0],
                },
                Stage::Relu { width: 2 },
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![0.0; 2],
                    wg: vec![1.0, -1.0],
                    bf: vec![0.0],
                    bg: vec![-0.25],
                },
                Stage::Relu { width: 1 },
            ],
        };
        let domain = BoxDomain {
            lo: vec![0.0; 2],
            hi: vec![1.0; 2],
        };
        let reference = refine::states(&chain, &domain, None, true).unwrap();
        let stronger =
            refine::states_strong_from_reference(&chain, &domain, None, true, &reference).unwrap();
        assert!(reference[3].g_lo[0] < 0.0 && reference[3].g_hi[0] > 0.0);
        assert!(stronger[3].g_hi[0] < 0.0);
        let baseline = build_with_cuts_until(&chain, &domain, None, false).unwrap();
        let strong = build(&chain, &domain).unwrap();
        assert_eq!(strong.bounds.len(), baseline.bounds.len());
        assert_eq!(strong.pivots, baseline.pivots);
        assert_eq!(strong.outputs, baseline.outputs);
        assert_eq!(strong.relu_pairs, baseline.relu_pairs);
        assert_eq!(strong.baseline_inequalities, baseline.inequalities.len());
        assert_eq!(strong.equalities.len(), baseline.equalities.len());
        for (a, b) in strong.equalities.iter().zip(&baseline.equalities) {
            assert_eq!(a.terms, b.terms);
            assert_eq!(a.rhs.to_bits(), b.rhs.to_bits());
        }
        for (a, b) in strong.inequalities.iter().zip(&baseline.inequalities) {
            assert_eq!(a.terms, b.terms);
            assert_eq!(a.rhs.to_bits(), b.rhs.to_bits());
        }
        assert!(strong
            .bounds
            .iter()
            .zip(&baseline.bounds)
            .all(|(&(l, u), &(bl, bu))| bl <= l && u <= bu));
        // No old free delta column disappears when G's final unit stabilizes.
        assert!(strong.inequalities.len() > baseline.inequalities.len());
        for i in 0..=8 {
            for j in 0..=8 {
                let (f, g) = chain.forward(&[i as f64 / 8.0, j as f64 / 8.0]).unwrap();
                assert_eq!(f[0] - g[0], 0.0);
            }
        }
    }

    #[test]
    fn timed_source_replay_preserves_rows_and_declines_expired_proofs() {
        let chain = chain();
        let domain = domain();
        let future = Instant::now() + Duration::from_secs(20);
        assert_eq!(chain.lp_dump_until(&domain, future), chain.lp_dump(&domain));
        let text = zero_certificate(&build(&chain, &domain).unwrap());
        assert!(
            chain
                .check_lp_text_until(&domain, 0.5, &text, future)
                .unwrap()
                .proved
        );
        assert!(
            chain
                .check_lp_texts_until(&domain, 0.5, &[&text, &text], future)
                .unwrap()
                .proved
        );
        let forged = text.clone() + " replacement_row";
        assert!(chain
            .check_lp_texts_until(&domain, 0.5, &[&text, &forged], future)
            .is_none());
        assert!(chain
            .check_lp_texts_until(&domain, 0.5, &[text.as_str(); 5], future)
            .is_none());
        let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        assert!(chain.lp_dump_until(&domain, past).is_none());
        assert!(chain
            .check_lp_text_until(&domain, 0.5, &text, past)
            .is_none());
    }

    #[test]
    fn authored_rows_enclose_every_dyadic_reachable_point() {
        let chain = chain();
        let model = build(&chain, &domain()).unwrap();
        for step in 0..=32 {
            let x = -1.0 + step as f64 / 16.0;
            let f = x.max(0.0);
            let g = (x + 0.25).max(0.0);
            let point = [x, x, x + 0.25, f, f - g, g];
            assert_eq!(model.bounds.len(), point.len());
            for (&value, &(l, h)) in point.iter().zip(&model.bounds) {
                assert!(value >= l && value <= h);
            }
            for row in &model.equalities {
                let lhs: f64 = row.terms.iter().map(|&(j, a)| a * point[j]).sum();
                assert_eq!(lhs, row.rhs);
            }
            for row in &model.inequalities {
                let lhs: f64 = row.terms.iter().map(|&(j, a)| a * point[j]).sum();
                assert!(lhs >= row.rhs, "{lhs} >= {}", row.rhs);
            }
        }
        assert_eq!(model.pivots, vec![1, 2, 5]);
        assert_eq!(model.outputs, vec![(3, 5)]);
    }

    #[test]
    fn multiplier_text_cannot_replace_source_rows_or_band() {
        let chain = chain();
        let domain = domain();
        let model = build(&chain, &domain).unwrap();
        let text = zero_certificate(&model);
        assert!(!chain.check_lp_text(&domain, 0.1, &text).unwrap().proved);
        assert!(chain.check_lp_text(&domain, 0.5, &text).unwrap().proved);
        assert!(chain
            .check_lp_text(&domain, 0.5, &(text.clone() + " changed_rhs"))
            .is_none());
        assert!(chain
            .check_lp_text(
                &domain,
                0.5,
                &text.replace("PAIRED_LP_DUAL_GEQ_V1", "PAIRED_LP_GEQ_V1")
            )
            .is_none());
        assert!(chain.check_lp_text(&domain, f64::NAN, &text).is_none());
        assert!(chain
            .check_lp_text(&domain, 0.5, &text.replace("0 ", "-1 "))
            .is_none());
        assert!(chain
            .check_lp_text(&domain, 0.5, &text.replace("0 ", "NaN "))
            .is_none());
    }

    #[test]
    fn signed_strip_dual_is_checked_against_exact_objective() {
        let chain = chain();
        let domain = domain();
        let model = build(&chain, &domain).unwrap();
        let mut text = format!(
            "PAIRED_LP_DUAL_GEQ_V1 {} {} 1\n",
            model.equalities.len(),
            model.inequalities.len()
        );
        for row_index in [
            model.baseline_inequalities - 2,
            model.baseline_inequalities - 1,
        ] {
            for _ in &model.equalities {
                text.push_str("0 ");
            }
            for i in 0..model.inequalities.len() {
                text.push_str(if i == row_index { "1 " } else { "0 " });
            }
        }
        let report = chain.check_lp_text(&domain, 0.5, &text).unwrap();
        assert!(report.proved);
        assert!(report.lower[0] <= -0.25 && report.lower[0] > -0.25000000000001);
        assert!(report.upper[0] >= 0.0 && report.upper[0] < 1e-12);
    }

    #[test]
    fn stable_relu_pivots_and_shared_inputs_are_exported_deterministically() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![Stage::Relu { width: 1 }],
        };
        let domain = BoxDomain {
            lo: vec![1.0],
            hi: vec![2.0],
        };
        let model = build(&chain, &domain).unwrap();
        assert_eq!(model.bounds.len(), 3);
        assert_eq!(model.pivots, vec![1, 2]);
        assert!(model.equalities.iter().all(|row| row.terms[1] == (0, -1.0)));
        let first = chain.lp_dump(&domain).unwrap();
        assert_eq!(first, chain.lp_dump(&domain).unwrap());
        assert!(first.starts_with(&format!(
            "PAIRED_LP_GEQ_V1 3 2 {} 1\n",
            model.inequalities.len()
        )));
        let baseline = chain
            .lp_dump_baseline_until(&domain, Instant::now() + Duration::from_secs(30))
            .unwrap();
        assert!(baseline.starts_with("PAIRED_LP_GEQ_V1 3 2 2 1\n"));
        assert!(
            chain
                .check_lp_text(&domain, 0.01, &zero_certificate(&model))
                .unwrap()
                .proved
        );
    }

    #[test]
    fn native_model_and_replay_deadlines_cannot_return_expired_evidence() {
        let chain = chain();
        let domain = domain();
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        assert!(build_until(&chain, &domain, Some(expired)).is_none());
        let model = build(&chain, &domain).unwrap();
        let dual = DualMultipliers {
            equality_multipliers: vec![0.0; model.equalities.len()],
            inequality_multipliers: vec![0.0; model.inequalities.len()],
        };
        let candidates = [dual.clone(), dual];
        assert!(check_candidates(&model, 0.5, &candidates, Some(expired)).is_none());
        assert!(
            check_candidates(&model, 0.5, &candidates, None)
                .unwrap()
                .proved
        );
        assert!(check_candidates(&model, 0.5, &candidates[..1], None).is_none());
    }

    #[test]
    fn complete_cover_root_cut_removes_independent_duplicate_relu_lift() {
        // Both F activations are authored from x; the difference is exactly
        // zero. G is zero. Scalar triangles permit an independent y1 lift.
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 2,
                    cols: 1,
                    wf: vec![1.0, 1.0],
                    wg: vec![0.0, 0.0],
                    bf: vec![0.0, 0.0],
                    bg: vec![0.0, 0.0],
                },
                Stage::Relu { width: 2 },
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![1.0, -1.0],
                    wg: vec![0.0, 0.0],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
            ],
        };
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let baseline = build_with_cuts_until(&chain, &domain, None, false).unwrap();
        let point = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0];
        assert_eq!(baseline.bounds.len(), point.len());
        assert!(baseline
            .bounds
            .iter()
            .zip(&point)
            .all(|(&(l, u), &x)| l <= x && x <= u));
        for row in &baseline.equalities {
            let value: f64 = row.terms.iter().map(|&(j, a)| a * point[j]).sum();
            assert_eq!(value, row.rhs);
        }
        for row in &baseline.inequalities {
            let value: f64 = row.terms.iter().map(|&(j, a)| a * point[j]).sum();
            assert!(value >= row.rhs, "baseline row {:?}", row);
        }
        let strong = build(&chain, &domain).unwrap();
        assert!(strong.inequalities[strong.baseline_inequalities..]
            .iter()
            .any(|row| { row.terms.iter().map(|&(j, a)| a * point[j]).sum::<f64>() < row.rhs }));
        assert!(strong.output_lo[0] <= 0.0 && strong.output_lo[0] > -1e-10);
        assert!(strong.output_hi[0] >= 0.0 && strong.output_hi[0] < 1e-10);
        let report = chain
            .check_lp_text(&domain, 1e-8, &zero_certificate(&strong))
            .unwrap();
        assert!(report.proved);
    }
}
