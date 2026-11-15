// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numerical equality elimination for AY proposals. The default full policy
//! preserves ALL source inequalities and pivot boxes as virtual GEQ rows.
//! The opt-in lean policy retains sparse/output rows and omits pivot-box
//! advice. Rounded substitution and row selection have no proof authority.
//! Source equality weights and source-row weights are lifted back and checked
//! against the unchanged original model, including its full residual box.

use super::*;

const MAX_FREE_VARIABLES: usize = 4096;
const MAX_DENSE_ENTRIES: usize = 40_000_000;
const MAX_REDUCED_NONZEROS: usize = 8_000_000;
const MAX_REDUCED_ROWS: usize = MAX_ROWS + 2 * MAX_VARIABLES;
const MAX_REDUCED_MAGNITUDE: f64 = 1e150;
const LEAN_MAX_SOURCE_NONZEROS: usize = 8;

/// Changes only AY's numerical advice. Every candidate is lifted into the
/// complete original source dimensions and checked against the source again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReducedProposalPolicy {
    #[default]
    Full,
    /// Retain authored GEQ rows with at most eight nonzero terms or an
    /// original output column. Original pivot boxes are omitted as advice.
    LeanSparseOutput,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReducedProposalStatistics {
    pub policy: ReducedProposalPolicy,
    pub selected_source_inequalities: usize,
    pub omitted_source_inequalities: usize,
    pub virtual_pivot_bound_rows: usize,
    pub omitted_virtual_pivot_bound_rows: usize,
    pub reduced_rows: usize,
    pub reduced_free_variables: usize,
    pub reduced_nonzeros: usize,
}

#[derive(Debug)]
pub struct ReducedProposalResult {
    pub result: AyResult,
    /// Absent when the worker declined or expired before completing reduction.
    pub statistics: Option<ReducedProposalStatistics>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReducedRole {
    Source(usize),
    PivotLower(usize),
    PivotUpper(usize),
}

#[cfg_attr(not(test), allow(dead_code))]
struct ReducedRow {
    terms: Vec<(usize, f64)>,
    rhs: f64,
    scale: f64,
    role: ReducedRole,
}

struct ReducedModel<'a> {
    // The borrowed immutable source also prevents a caller from mutating its
    // rows between reduction and lifting. All mappings belong to THIS source.
    source: &'a SourceModel,
    model: Model,
    columns: Vec<Col>,
    rows: Vec<ReducedRow>,
    // Row-major approximate x = P*q+h, q in [-1,1]. These expressions are
    // advice only; no projected primal or LP-optimum claim is made.
    expressions: Vec<f64>,
    stride: usize,
    pivots: Vec<(usize, usize)>,
    free: usize,
    nonzeros: usize,
    statistics: ReducedProposalStatistics,
}

fn finite(x: f64) -> Result<f64, String> {
    if x.is_finite() && x.abs() <= MAX_REDUCED_MAGNITUDE {
        Ok(x)
    } else {
        Err("reduced numerical coefficient cap".into())
    }
}

fn admit_bytes(bytes: usize, budget: usize) -> Result<(), String> {
    if bytes > budget {
        Err(format!("reduced workspace admission {bytes} > {budget}"))
    } else {
        Ok(())
    }
}

impl ReducedModel<'_> {
    fn check_objective(
        &self,
        source: &SourceModel,
        output: usize,
        sign: f64,
    ) -> Result<(), String> {
        if !std::ptr::eq(self.source, source) {
            return Err("reduced source binding mismatch".into());
        }
        if output >= source.outputs.len() || (sign != 1.0 && sign != -1.0) {
            return Err("reduced objective binding".into());
        }
        Ok(())
    }

    fn append(
        &mut self,
        coefficients: &[f64],
        rhs: f64,
        role: ReducedRole,
        memory_bytes: usize,
        deadline: Instant,
    ) -> Result<(), String> {
        expired(deadline)?;
        let mut norm = 0.0;
        for (j, &a) in coefficients.iter().enumerate() {
            if j & 0x3ff == 0 {
                expired(deadline)?;
            }
            norm = finite(norm + finite(a)?.abs())?;
        }
        let rhs = finite(rhs)?;
        // Both row coefficients and RHS are preconditioned. This division is
        // numerical advice; the source checker never receives these rows.
        let scale = norm.max(rhs.abs()).max(1e-12);
        let terms: Vec<_> = coefficients
            .iter()
            .enumerate()
            .filter_map(|(j, &a)| (a != 0.0).then_some((j, a / scale)))
            .collect();
        self.nonzeros = self
            .nonzeros
            .checked_add(terms.len())
            .ok_or("reduced nonzero overflow")?;
        if self.nonzeros > MAX_REDUCED_NONZEROS {
            return Err("reduced nonzero cap".into());
        }
        let rows = self.rows.len() + 1;
        if rows > MAX_REDUCED_ROWS {
            return Err("reduced row cap".into());
        }
        let bytes = self
            .expressions
            .len()
            .checked_mul(8)
            .and_then(|n| n.checked_add(self.free.checked_mul(rows)?.checked_mul(8)?))
            .and_then(|n| n.checked_add(self.nonzeros.checked_mul(384)?))
            .and_then(|n| n.checked_add(32 * 1024 * 1024))
            .ok_or("reduced memory estimate overflow")?;
        admit_bytes(bytes, memory_bytes)?;
        let ay_terms: Vec<_> = terms.iter().map(|&(j, a)| (self.columns[j], a)).collect();
        let row = self.model.add_row(rhs / scale, f64::INFINITY, &ay_terms);
        if row.index() != self.rows.len() {
            return Err("reduced AY row order".into());
        }
        self.rows.push(ReducedRow {
            terms,
            rhs: rhs / scale,
            scale,
            role,
        });
        Ok(())
    }

    fn objective(
        &self,
        source: &SourceModel,
        output: usize,
        sign: f64,
    ) -> Result<(Vec<(Col, f64)>, f64), String> {
        self.check_objective(source, output, sign)?;
        let (f, g) = source.outputs[output];
        let mut coefficients = Vec::with_capacity(self.free);
        let mut norm = 0.0;
        for j in 0..self.free {
            let a = finite(
                sign * (self.expressions[f * self.stride + j]
                    - self.expressions[g * self.stride + j]),
            )?;
            norm = finite(norm + a.abs())?;
            coefficients.push(a);
        }
        let scale = norm.max(1e-12);
        let terms = coefficients
            .into_iter()
            .enumerate()
            .filter_map(|(j, a)| (a != 0.0).then_some((self.columns[j], a / scale)))
            .collect();
        Ok((terms, scale))
    }

    /// Ordinary float Lagrangian score on the approximate reduced rows. This
    /// diagnoses proposal conditioning only; it has no optimum/proof authority.
    fn numerical_lower(
        &self,
        source: &SourceModel,
        raw: &[f64],
        output: usize,
        sign: f64,
        convention: f64,
        objective_scale: f64,
        deadline: Instant,
    ) -> Option<f64> {
        if self.check_objective(source, output, sign).is_err()
            || raw.len() != self.rows.len()
            || (convention != 1.0 && convention != -1.0)
            || !objective_scale.is_finite()
            || objective_scale <= 0.0
        {
            return None;
        }
        let (f, g) = source.outputs[output];
        let mut residual: Vec<_> = (0..self.free)
            .map(|j| {
                sign * (self.expressions[f * self.stride + j]
                    - self.expressions[g * self.stride + j])
                    / objective_scale
            })
            .collect();
        let mut value = 0.0;
        for (&raw, row) in raw.iter().zip(&self.rows) {
            expired(deadline).ok()?;
            if !raw.is_finite() {
                return None;
            }
            let weight = (raw * convention).max(0.0);
            value += weight * row.rhs;
            for (k, &(j, a)) in row.terms.iter().enumerate() {
                if k & 0x3ff == 0 {
                    expired(deadline).ok()?;
                }
                residual[j] -= weight * a;
            }
        }
        // Every approximate free coordinate has exactly the advice box [-1,1].
        value -= residual.iter().map(|x| x.abs()).sum::<f64>();
        value = value * objective_scale
            + sign
                * (self.expressions[f * self.stride + self.free]
                    - self.expressions[g * self.stride + self.free]);
        expired(deadline).ok()?;
        value.is_finite().then_some(value)
    }

    fn lift(
        &self,
        source: &SourceModel,
        raw: &[f64],
        output: usize,
        sign: f64,
        convention: f64,
        objective_scale: f64,
        deadline: Instant,
    ) -> Result<RoundedDual, String> {
        expired(deadline)?;
        self.check_objective(source, output, sign)?;
        if raw.len() != self.rows.len()
            || (convention != 1.0 && convention != -1.0)
            || !objective_scale.is_finite()
            || objective_scale <= 0.0
        {
            return Err("reduced dual arity/convention".into());
        }
        let mut dual = source.zero_dual();
        let mut beta = vec![0.0; source.bounds.len()];
        for (&y, row) in raw.iter().zip(&self.rows) {
            expired(deadline)?;
            if !y.is_finite() {
                return Err("nonfinite reduced dual".into());
            }
            let weight = finite((y * convention).max(0.0) * (objective_scale / row.scale))?;
            if weight > MAX_MAGNITUDE {
                return Err("lifted multiplier cap".into());
            }
            match row.role {
                ReducedRole::Source(i) => dual.inequalities[i] = weight,
                ReducedRole::PivotLower(p) => beta[p] = finite(beta[p] + weight)?,
                ReducedRole::PivotUpper(p) => beta[p] = finite(beta[p] - weight)?,
            }
        }
        // Keep the boxed pivot contribution instead of incorrectly forcing
        // every pivot residual to zero. All rounding is independently replayed.
        let mut residual = vec![0.0; source.bounds.len()];
        let (f, g) = source.outputs[output];
        residual[f] += sign;
        residual[g] -= sign;
        for (row, &weight) in source.inequalities.iter().zip(&dual.inequalities) {
            expired(deadline)?;
            if weight == 0.0 {
                continue;
            }
            for (k, &(j, a)) in row.terms.iter().enumerate() {
                if k & 0x3ff == 0 {
                    expired(deadline)?;
                }
                residual[j] = finite(residual[j] - weight * a)?;
            }
        }
        for &(p, i) in self.pivots.iter().rev() {
            expired(deadline)?;
            let mu = finite(residual[p] - beta[p])?;
            if mu.abs() > MAX_MAGNITUDE {
                return Err("lifted equality multiplier cap".into());
            }
            dual.equalities[i] = mu;
            for (k, &(j, a)) in source.equalities[i].terms.iter().enumerate() {
                if k & 0x3ff == 0 {
                    expired(deadline)?;
                }
                if j != p {
                    residual[j] = finite(residual[j] - mu * a)?;
                }
            }
            residual[p] = beta[p];
        }
        expired(deadline)?;
        Ok(dual)
    }
}

fn build(
    source: &SourceModel,
    deadline: Instant,
    memory_bytes: usize,
) -> Result<ReducedModel<'_>, String> {
    build_with_policy(source, deadline, memory_bytes, ReducedProposalPolicy::Full)
}

fn select_source_row(row: &Row, outputs: &BTreeSet<usize>, policy: ReducedProposalPolicy) -> bool {
    policy == ReducedProposalPolicy::Full
        || row.terms.iter().filter(|&&(_, a)| a != 0.0).count() <= LEAN_MAX_SOURCE_NONZEROS
        || row
            .terms
            .iter()
            .any(|&(j, a)| a != 0.0 && outputs.contains(&j))
}

fn build_with_policy(
    source: &SourceModel,
    deadline: Instant,
    memory_bytes: usize,
    policy: ReducedProposalPolicy,
) -> Result<ReducedModel<'_>, String> {
    expired(deadline)?;
    if source.pivots.len() != source.equalities.len() {
        return Err("source pivot arity".into());
    }
    let mut pivots: Vec<_> = source
        .pivots
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();
    pivots.sort_unstable_by_key(|&(p, _)| p);
    let pivot_set: BTreeSet<_> = source.pivots.iter().copied().collect();
    let free_columns: Vec<_> = source
        .bounds
        .iter()
        .enumerate()
        .filter_map(|(j, &(l, h))| (!pivot_set.contains(&j) && l < h).then_some(j))
        .collect();
    let free = free_columns.len();
    if free > MAX_FREE_VARIABLES {
        return Err("reduced free-variable cap".into());
    }
    let stride = free + 1;
    let entries = source
        .bounds
        .len()
        .checked_mul(stride)
        .ok_or("reduced dense dimension overflow")?;
    if entries > MAX_DENSE_ENTRIES {
        return Err("reduced dense-expression cap".into());
    }
    let source_terms: usize = source
        .equalities
        .iter()
        .chain(&source.inequalities)
        .try_fold(0usize, |n, r| n.checked_add(r.terms.len()))
        .ok_or("reduced source term overflow")?;
    let preliminary = entries
        .checked_mul(8)
        .and_then(|n| n.checked_add(source_terms.checked_mul(32)?))
        .and_then(|n| n.checked_add(32 * 1024 * 1024))
        .ok_or("reduced preliminary memory overflow")?;
    admit_bytes(preliminary, memory_bytes)?;
    let mut expressions = vec![0.0; entries];
    for (j, &(l, _)) in source.bounds.iter().enumerate() {
        if !pivot_set.contains(&j) {
            expressions[j * stride + free] = l;
        }
    }
    for (k, &j) in free_columns.iter().enumerate() {
        let (l, h) = source.bounds[j];
        let center = finite(0.5 * l + 0.5 * h)?;
        let radius = finite(0.5 * h - 0.5 * l)?;
        if radius <= 0.0 {
            return Err("reduced radius underflow".into());
        }
        expressions[j * stride + k] = radius;
        expressions[j * stride + free] = center;
    }
    for &(p, i) in &pivots {
        expired(deadline)?;
        let row = &source.equalities[i];
        expressions[p * stride + free] = row.rhs;
        for (entry, &(j, a)) in row.terms.iter().enumerate() {
            if entry & 0x3f == 0 {
                expired(deadline)?;
            }
            if j == p {
                continue;
            }
            for k in 0..stride {
                if k & 0x3ff == 0 {
                    expired(deadline)?;
                }
                expressions[p * stride + k] =
                    finite(expressions[p * stride + k] - a * expressions[j * stride + k])?;
            }
        }
    }
    let mut model = Model::new();
    let columns = (0..free).map(|_| model.add_col(-1.0, 1.0)).collect();
    let mut result = ReducedModel {
        source,
        model,
        columns,
        rows: Vec::new(),
        expressions,
        stride,
        pivots,
        free,
        nonzeros: 0,
        statistics: ReducedProposalStatistics {
            policy,
            selected_source_inequalities: 0,
            omitted_source_inequalities: 0,
            virtual_pivot_bound_rows: 0,
            omitted_virtual_pivot_bound_rows: 0,
            reduced_rows: 0,
            reduced_free_variables: free,
            reduced_nonzeros: 0,
        },
    };
    let outputs: BTreeSet<_> = source.outputs.iter().flat_map(|&(f, g)| [f, g]).collect();
    let mut coefficients = vec![0.0; free];
    for (i, row) in source.inequalities.iter().enumerate() {
        expired(deadline)?;
        if !select_source_row(row, &outputs, policy) {
            result.statistics.omitted_source_inequalities += 1;
            continue;
        }
        coefficients.fill(0.0);
        let mut rhs = row.rhs;
        for (entry, &(j, a)) in row.terms.iter().enumerate() {
            if entry & 0x3f == 0 {
                expired(deadline)?;
            }
            rhs = finite(rhs - a * result.expressions[j * stride + free])?;
            for k in 0..free {
                if k & 0x3ff == 0 {
                    expired(deadline)?;
                }
                coefficients[k] = finite(coefficients[k] + a * result.expressions[j * stride + k])?;
            }
        }
        result.append(
            &coefficients,
            rhs,
            ReducedRole::Source(i),
            memory_bytes,
            deadline,
        )?;
        result.statistics.selected_source_inequalities += 1;
    }
    // Full keeps BOTH original bound facts for every eliminated target. Lean
    // omits them only as advice: beta stays zero, and ORIGINAL source replay
    // still prices every residual against every original pivot box.
    // Free-column boxes remain normalized AY variable boxes in both policies.
    let virtual_rows = result
        .pivots
        .len()
        .checked_mul(2)
        .ok_or("pivot-box row overflow")?;
    let retained_pivots = if policy == ReducedProposalPolicy::Full {
        result.statistics.virtual_pivot_bound_rows = virtual_rows;
        result.pivots.len()
    } else {
        result.statistics.omitted_virtual_pivot_bound_rows = virtual_rows;
        0
    };
    for index in 0..retained_pivots {
        let (p, _) = result.pivots[index];
        let (l, h) = source.bounds[p];
        coefficients.copy_from_slice(&result.expressions[p * stride..p * stride + free]);
        let constant = result.expressions[p * stride + free];
        result.append(
            &coefficients,
            finite(l - constant)?,
            ReducedRole::PivotLower(p),
            memory_bytes,
            deadline,
        )?;
        for a in &mut coefficients {
            *a = -*a;
        }
        result.append(
            &coefficients,
            finite(constant - h)?,
            ReducedRole::PivotUpper(p),
            memory_bytes,
            deadline,
        )?;
    }
    result.statistics.reduced_rows = result.rows.len();
    result.statistics.reduced_nonzeros = result.nonzeros;
    expired(deadline)?;
    Ok(result)
}

/// Numeric comparison for an archived ORIGINAL-row multiplier transcript.
/// The canonical virtual rows use just the net signed pivot residual; opposing
/// virtual weights canceled by the original solve cannot be reconstructed.
/// This report authenticates no proof, optimum, or reduced primal point.
#[derive(Clone, Debug, Serialize)]
pub struct NumericalReductionReplay {
    pub output: usize,
    pub sign: i32,
    pub numerical_source_lower: f64,
    pub canonical_reduced_lower: f64,
    pub substitution_gap: f64,
}

pub fn diagnose_reduced_dual_text(
    source: &SourceModel,
    text: &str,
    deadline: Instant,
    memory_bytes: usize,
) -> Result<Vec<NumericalReductionReplay>, String> {
    if text.len() > MAX_BYTES
        || !(16 * 1024 * 1024..=2 * 1024 * 1024 * 1024).contains(&memory_bytes)
    {
        return Err("reduced diagnostic admission cap".into());
    }
    expired(deadline)?;
    let reduced = build(source, deadline, memory_bytes)?;
    let mut tokens = text.split_whitespace();
    if tokens.next() != Some("PAIRED_LP_DUAL_GEQ_V1") {
        return Err("reduced diagnostic dual sense/version".into());
    }
    for expected in [
        source.equalities.len(),
        source.inequalities.len(),
        source.outputs.len(),
    ] {
        let value = tokens
            .next()
            .ok_or("missing dual dimension")?
            .parse::<usize>()
            .map_err(|_| "bad dual dimension")?;
        if value != expected {
            return Err("reduced diagnostic dual dimension".into());
        }
    }
    let mut reports = Vec::with_capacity(2 * source.outputs.len());
    for output in 0..source.outputs.len() {
        for sign in [1.0, -1.0] {
            expired(deadline)?;
            let mut dual = source.zero_dual();
            for (nonnegative, weights) in [
                (false, &mut dual.equalities),
                (true, &mut dual.inequalities),
            ] {
                for (i, weight) in weights.iter_mut().enumerate() {
                    if i & 0x3ff == 0 {
                        expired(deadline)?;
                    }
                    *weight = tokens
                        .next()
                        .ok_or("missing dual multiplier")?
                        .parse::<f64>()
                        .map_err(|_| "bad dual multiplier")?;
                    if !weight.is_finite()
                        || weight.abs() > MAX_MAGNITUDE
                        || (nonnegative && *weight < 0.0)
                    {
                        return Err("reduced diagnostic multiplier cap/sign".into());
                    }
                }
            }
            let mut residual = vec![0.0; source.bounds.len()];
            let (f, g) = source.outputs[output];
            residual[f] += sign;
            residual[g] -= sign;
            for (rows, weights) in [
                (&source.equalities, &dual.equalities),
                (&source.inequalities, &dual.inequalities),
            ] {
                for (row, &weight) in rows.iter().zip(weights) {
                    expired(deadline)?;
                    if weight == 0.0 {
                        continue;
                    }
                    for (k, &(j, a)) in row.terms.iter().enumerate() {
                        if k & 0x3ff == 0 {
                            expired(deadline)?;
                        }
                        residual[j] = finite(residual[j] - weight * a)?;
                    }
                }
            }
            let (_, scale) = reduced.objective(source, output, sign)?;
            let mut raw = Vec::with_capacity(reduced.rows.len());
            for row in &reduced.rows {
                expired(deadline)?;
                let weight = match row.role {
                    ReducedRole::Source(i) => dual.inequalities[i],
                    ReducedRole::PivotLower(p) => residual[p].max(0.0),
                    ReducedRole::PivotUpper(p) => (-residual[p]).max(0.0),
                };
                raw.push(finite(weight * (row.scale / scale))?);
            }
            let original = numerical_score_until(source, &dual, output, sign, Some(deadline))
                .ok_or("source numeric diagnostic declined")?;
            let canonical = reduced
                .numerical_lower(source, &raw, output, sign, 1.0, scale, deadline)
                .ok_or("reduced numeric diagnostic declined")?;
            let gap = finite(original - canonical)?;
            reports.push(NumericalReductionReplay {
                output,
                sign: sign as i32,
                numerical_source_lower: original,
                canonical_reduced_lower: canonical,
                substitution_gap: gap,
            });
        }
    }
    if tokens.next().is_some() {
        return Err("trailing diagnostic dual text".into());
    }
    expired(deadline)?;
    Ok(reports)
}

/// Preconditioned numerical AY proposal path. The approximate reduced LP has
/// no authority and supplies no primal/optimum bracket. Both lifted sign
/// conventions are returned for original-source residual replay.
pub fn solve_reduced_float(
    source: &SourceModel,
    deadline: Instant,
    memory_bytes: usize,
) -> Result<AyResult, String> {
    solve_reduced_float_with_policy(source, deadline, memory_bytes, ReducedProposalPolicy::Full)
        .map(|r| r.result)
}

/// Opt-in row thinning changes only the numerical proposal. Omitted source
/// inequalities lift to exact zero weights in the original-length vector;
/// all authored equalities and all original boxes remain replay authority.
pub fn solve_reduced_float_with_policy(
    source: &SourceModel,
    deadline: Instant,
    memory_bytes: usize,
    policy: ReducedProposalPolicy,
) -> Result<ReducedProposalResult, String> {
    expired(deadline)?;
    if !(16 * 1024 * 1024..=2 * 1024 * 1024 * 1024).contains(&memory_bytes) {
        return Err("AY reduced memory budget out of range".into());
    }
    let permit = acquire_worker(&ACTIVE_WORKERS, MAX_ACTIVE_WORKERS)?;
    let owned = source.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    let (statistics_sender, statistics_receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("ny-paired-ay-reduced".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let _permit = permit;
            let work = || -> Result<(), String> {
                let reduced = build_with_policy(&owned, deadline, memory_bytes, policy)?;
                let _ = statistics_sender.send(reduced.statistics.clone());
                let _ = sender.send(Event::ReducedReady(reduced.free, reduced.nonzeros));
                let options = solver_options(deadline, memory_bytes)
                    .with_engine(ay_milp::EngineEconomics::new().with_float_lane(true));
                let mut session =
                    LpSession::new(&reduced.model, &options).map_err(|e| e.to_string())?;
                for output in 0..owned.outputs.len() {
                    for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
                        expired(deadline)?;
                        let start = Instant::now();
                        let (objective, objective_scale) =
                            reduced.objective(&owned, output, sign)?;
                        let raw = if objective.is_empty() {
                            Ok(Some(vec![0.0; reduced.rows.len()]))
                        } else {
                            session.float_dual_for_expr(&objective, Sense::Minimize)
                        };
                        let result = raw.map_err(|e| e.to_string()).and_then(|raw| {
                            let raw = raw.ok_or("AY reduced float simplex declined proposal")?;
                            let positive = reduced.lift(
                                &owned,
                                &raw,
                                output,
                                sign,
                                1.0,
                                objective_scale,
                                deadline,
                            )?;
                            let negative = reduced.lift(
                                &owned,
                                &raw,
                                output,
                                sign,
                                -1.0,
                                objective_scale,
                                deadline,
                            )?;
                            let ps = numerical_score_until(
                                &owned,
                                &positive,
                                output,
                                sign,
                                Some(deadline),
                            );
                            let ns = numerical_score_until(
                                &owned,
                                &negative,
                                output,
                                sign,
                                Some(deadline),
                            );
                            let prefer_negative = ns.is_some_and(|n| ps.is_none_or(|p| n > p));
                            let reduced_score = reduced.numerical_lower(
                                &owned,
                                &raw,
                                output,
                                sign,
                                if prefer_negative { -1.0 } else { 1.0 },
                                objective_scale,
                                deadline,
                            );
                            Ok(if prefer_negative {
                                (negative, positive, ns, reduced_score)
                            } else {
                                (positive, negative, ps, reduced_score)
                            })
                        });
                        let mut diagnostic = diagnostic(output, sign, start);
                        match result {
                            Ok((first, second, score, reduced_score)) => {
                                diagnostic.status = "untrusted_reduced_float_duals".into();
                                diagnostic.numerical_proposal_lower = score;
                                diagnostic.reduced_proposal_lower = reduced_score;
                                diagnostic.reduction_lift_gap = score
                                    .zip(reduced_score)
                                    .map(|(original, reduced)| original - reduced)
                                    .filter(|x| x.is_finite());
                                expired(deadline)?;
                                sender
                                    .send(Event::Objective(Box::new((
                                        2 * output + side,
                                        first,
                                        Some(second),
                                        diagnostic,
                                    ))))
                                    .map_err(|_| "caller expired".to_owned())?;
                            }
                            Err(reason) => {
                                diagnostic.status = reason;
                                let _ = sender.send(Event::Failure(Box::new(diagnostic)));
                            }
                        }
                    }
                }
                Ok(())
            };
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
                Ok(Err(reason)) => {
                    let _ = sender.send(Event::Termination(reason));
                }
                Err(_) => {
                    let _ = sender.send(Event::Termination("AY reduced worker panic".into()));
                }
                Ok(Ok(())) => {}
            }
            let _ = sender.send(Event::Finished);
        })
        .map_err(|e| e.to_string())?;
    let result = collect(source, deadline, receiver);
    Ok(ReducedProposalResult {
        result,
        statistics: statistics_receiver.try_recv().ok(),
    })
}

#[cfg(test)]
#[path = "paired_difference_ay_reduced_tests.rs"]
mod tests;
