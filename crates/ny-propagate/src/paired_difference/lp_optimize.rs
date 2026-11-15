// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! First-party numerical dual proposals for the authenticated paired LP.
//!
//! Authored affine equalities are eliminated only inside this optimizer.
//! Nonnegative inequality multipliers are seeded by reverse substitution and
//! improved with projected Adam on the box-residual dual. The result contains
//! multipliers for the ORIGINAL rows. Nothing in this module certifies a
//! bound: `lp_certificate::certified_lower` must replay every result, including
//! its complete residual, against the source-built model.
//!
//! Nearest-rounded reduction, normalization, smoothing and stopping scores
//! are deliberately outside the proof authority. This module uses only Rust's
//! standard library, retains no foreign solver rows, and launches no process.

use super::lp_certificate::{DualMultipliers, LinearRow};
use super::lp_model::Model;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const MAX_DENSE_ENTRIES: usize = 40_000_000;
const MAX_REDUCED_NONZEROS: usize = 8_000_000;
const MAX_OBJECTIVES: usize = 128;
const MAX_MULTIPLIER: f64 = 1e100;

/// Numerical controls only. A stopping target does not authorize a proof.
#[derive(Clone, Copy, Debug)]
pub struct OptimizeOptions {
    pub steps: usize,
    pub learning_rate: f64,
    pub anneal: f64,
    pub workers: usize,
    pub target_lower: Option<f64>,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            steps: 10_000,
            learning_rate: 0.01,
            anneal: 0.995,
            workers: 8,
            target_lower: None,
        }
    }
}

impl OptimizeOptions {
    fn valid(self) -> bool {
        self.steps <= 100_000
            && self.learning_rate.is_finite()
            && self.learning_rate > 0.0
            && self.anneal.is_finite()
            && self.anneal > 0.0
            && self.anneal <= 1.0
            && (1..=16).contains(&self.workers)
            && self.target_lower.is_none_or(f64::is_finite)
    }
}

#[derive(Clone, Debug)]
pub(super) struct Candidate {
    pub(super) output: usize,
    pub(super) sign: f64,
    pub(super) dual: DualMultipliers,
    pub(super) proposal_lower: f64,
    pub(super) iterations: usize,
}

#[derive(Clone, Debug)]
pub(super) struct OptimizationResult {
    /// Exactly two entries per output: positive sign followed by negative.
    pub(super) candidates: Vec<Candidate>,
    pub(super) free_variables: usize,
    pub(super) reduced_nonzeros: usize,
}

struct SparseRows {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    coefficients: Vec<f64>,
    /// Group rows by their largest remaining variable for reverse seeding.
    ending_at: Vec<Vec<usize>>,
}

impl SparseRows {
    fn terms(&self, row: usize) -> std::ops::Range<usize> {
        self.offsets[row]..self.offsets[row + 1]
    }
}

struct Reduced {
    free: Vec<usize>,
    ordered_equalities: Vec<usize>,
    rows: SparseRows,
    rhs: Vec<f64>,
    scales: Vec<f64>,
    objectives: Vec<Vec<f64>>,
    constants: Vec<f64>,
}

fn zeroes(n: usize) -> Option<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(n).ok()?;
    values.resize(n, 0.0);
    Some(values)
}

fn expired(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

fn finite(values: &[f64]) -> bool {
    values.iter().all(|x| x.is_finite())
}

fn admissible_row(row: &LinearRow, variables: usize) -> bool {
    row.rhs.is_finite()
        && row
            .terms
            .iter()
            .all(|&(column, coefficient)| column < variables && coefficient.is_finite())
}

/// Eliminate authored +1 pivots, preserving the original rows in `model`.
fn reduce(model: &Model, deadline: Instant) -> Option<Reduced> {
    let variables = model.bounds.len();
    let objective_count = model.outputs.len().checked_mul(2)?;
    if variables == 0
        || objective_count == 0
        || objective_count > MAX_OBJECTIVES
        || model.pivots.len() != model.equalities.len()
        || model
            .bounds
            .iter()
            .any(|&(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        || model
            .equalities
            .iter()
            .chain(&model.inequalities)
            .any(|row| !admissible_row(row, variables))
        || model
            .outputs
            .iter()
            .any(|&(f, g)| f >= variables || g >= variables)
        || expired(deadline)
    {
        return None;
    }
    let mut pivot_rows = vec![None; variables];
    for (index, (&pivot, row)) in model.pivots.iter().zip(&model.equalities).enumerate() {
        if pivot >= variables
            || pivot_rows[pivot].is_some()
            || row.terms.iter().filter(|&&(j, _)| j == pivot).count() != 1
            || !row.terms.iter().any(|&(j, a)| j == pivot && a == 1.0)
            || row.terms.iter().any(|&(j, _)| j > pivot)
        {
            return None;
        }
        pivot_rows[pivot] = Some(index);
    }
    let free: Vec<_> = pivot_rows
        .iter()
        .enumerate()
        .filter_map(|(j, row)| row.is_none().then_some(j))
        .collect();
    let width = free.len();
    let dense_entries = variables.checked_mul(width)?;
    if dense_entries > MAX_DENSE_ENTRIES
        || model.inequalities.len().checked_mul(width)? > MAX_DENSE_ENTRIES
    {
        return None;
    }
    let mut transform = zeroes(dense_entries)?;
    let mut offset = zeroes(variables)?;
    for (j, &original) in free.iter().enumerate() {
        transform[original * width + j] = 1.0;
    }
    let mut ordered_equalities = Vec::with_capacity(model.equalities.len());
    for (pivot, row_index) in pivot_rows.iter().enumerate() {
        let Some(row_index) = *row_index else {
            continue;
        };
        if expired(deadline) {
            return None;
        }
        let row = &model.equalities[row_index];
        ordered_equalities.push(row_index);
        let mut intercept = row.rhs;
        let (earlier, target_and_later) = transform.split_at_mut(pivot * width);
        let target = &mut target_and_later[..width];
        for (term_index, &(source, coefficient)) in row.terms.iter().enumerate() {
            if term_index % 64 == 0 && expired(deadline) {
                return None;
            }
            if source == pivot {
                continue;
            }
            intercept -= coefficient * offset[source];
            for (a, &b) in target
                .iter_mut()
                .zip(&earlier[source * width..(source + 1) * width])
            {
                *a -= coefficient * b;
            }
        }
        if !intercept.is_finite() || !finite(target) {
            return None;
        }
        offset[pivot] = intercept;
    }
    let center: Vec<_> = free
        .iter()
        .map(|&j| model.bounds[j].0 * 0.5 + model.bounds[j].1 * 0.5)
        .collect();
    let radius: Vec<_> = free
        .iter()
        .map(|&j| model.bounds[j].1 * 0.5 - model.bounds[j].0 * 0.5)
        .collect();
    let mut rows = SparseRows {
        offsets: vec![0],
        columns: Vec::new(),
        coefficients: Vec::new(),
        ending_at: vec![Vec::new(); width],
    };
    let mut reduced_rhs = Vec::with_capacity(model.inequalities.len());
    let mut scales = Vec::with_capacity(model.inequalities.len());
    let mut reduced_row = zeroes(width)?;
    for (row_index, row) in model.inequalities.iter().enumerate() {
        if expired(deadline) {
            return None;
        }
        reduced_row.fill(0.0);
        let mut rhs = row.rhs;
        // Repeated source columns intentionally accumulate. The original
        // model/checker retains those repeated terms without rewriting them.
        for (term_index, &(column, coefficient)) in row.terms.iter().enumerate() {
            if term_index % 64 == 0 && expired(deadline) {
                return None;
            }
            rhs -= coefficient * offset[column];
            for (a, &b) in reduced_row
                .iter_mut()
                .zip(&transform[column * width..(column + 1) * width])
            {
                *a += coefficient * b;
            }
        }
        let mut norm = 0.0;
        for j in 0..width {
            rhs -= reduced_row[j] * center[j];
            reduced_row[j] *= radius[j];
            norm += reduced_row[j].abs();
        }
        let scale = norm.max(rhs.abs()).max(1e-12);
        if !rhs.is_finite() || !scale.is_finite() || !finite(&reduced_row) {
            return None;
        }
        let nnz = reduced_row.iter().filter(|&&v| v != 0.0).count();
        if rows.columns.len().checked_add(nnz)? > MAX_REDUCED_NONZEROS {
            return None;
        }
        rows.columns.try_reserve(nnz).ok()?;
        rows.coefficients.try_reserve(nnz).ok()?;
        for (column, &value) in reduced_row.iter().enumerate() {
            let value = value / scale;
            if value != 0.0 {
                rows.columns.push(column);
                rows.coefficients.push(value);
            }
        }
        if rows.columns.len() > *rows.offsets.last()? {
            rows.ending_at[*rows.columns.last()?].push(row_index);
        }
        rows.offsets.push(rows.columns.len());
        reduced_rhs.push(rhs / scale);
        scales.push(scale);
    }
    let mut objectives = Vec::with_capacity(objective_count);
    let mut constants = Vec::with_capacity(objective_count);
    for &(f, g) in &model.outputs {
        if expired(deadline) {
            return None;
        }
        let mut objective = zeroes(width)?;
        let mut constant = offset[f] - offset[g];
        for j in 0..width {
            let coefficient = transform[f * width + j] - transform[g * width + j];
            constant += coefficient * center[j];
            objective[j] = coefficient * radius[j];
        }
        if !constant.is_finite() || !finite(&objective) {
            return None;
        }
        objectives.push(objective.clone());
        objectives.push(objective.into_iter().map(|v| -v).collect());
        constants.extend([constant, -constant]);
    }
    Some(Reduced {
        free,
        ordered_equalities,
        rows,
        rhs: reduced_rhs,
        scales,
        objectives,
        constants,
    })
}

/// Actual nonsmooth numerical score, never the smoothed optimization score.
fn dual_value(
    reduced: &Reduced,
    objective: usize,
    weights: &[f64],
    residual: &mut [f64],
) -> Option<f64> {
    residual.copy_from_slice(&reduced.objectives[objective]);
    let mut linear = reduced.constants[objective];
    for (row, &weight) in weights.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        if !weight.is_finite() || weight < 0.0 {
            return None;
        }
        linear += weight * reduced.rhs[row];
        for term in reduced.rows.terms(row) {
            residual[reduced.rows.columns[term]] -= weight * reduced.rows.coefficients[term];
        }
    }
    let value = linear - residual.iter().map(|v| v.abs()).sum::<f64>();
    (value.is_finite() && finite(residual)).then_some(value)
}

/// Choose nonnegative reverse-substitution planes that improve the real
/// nonsmooth proposal objective. Largest-free-column grouping works for both
/// independent ReLU rows and coupled signed supports.
fn backward_seed(
    reduced: &Reduced,
    objective: usize,
    deadline: Instant,
    force: bool,
) -> Option<Vec<f64>> {
    let mut weights = zeroes(reduced.rhs.len())?;
    let mut residual = reduced.objectives[objective].clone();
    let mut linear = reduced.constants[objective];
    for target in (0..reduced.free.len()).rev() {
        if expired(deadline) {
            break;
        }
        if residual[target] == 0.0 {
            continue;
        }
        let current = linear - residual.iter().map(|v| v.abs()).sum::<f64>();
        // Full reverse substitution can cross a locally weaker intermediate
        // box bound before recovering a stronger bound at the shared input.
        // Keep it as an additional proposal; compare complete dual values.
        let mut best_value = if force { f64::NEG_INFINITY } else { current };
        let mut best = None;
        for (position, &row) in reduced.rows.ending_at[target].iter().enumerate() {
            if position % 64 == 0 && expired(deadline) {
                break;
            }
            let last = reduced.rows.offsets[row + 1] - 1;
            let weight = residual[target] / reduced.rows.coefficients[last];
            if !weight.is_finite() || weight <= 0.0 || weight / reduced.scales[row] > MAX_MULTIPLIER
            {
                continue;
            }
            let mut value = current + weight * reduced.rhs[row];
            for term in reduced.rows.terms(row) {
                let before = residual[reduced.rows.columns[term]];
                value += before.abs() - (before - weight * reduced.rows.coefficients[term]).abs();
            }
            if value.is_finite() && value >= best_value {
                best_value = value;
                best = Some((row, weight));
            }
        }
        if let Some((row, weight)) = best {
            weights[row] += weight;
            linear += weight * reduced.rhs[row];
            for term in reduced.rows.terms(row) {
                residual[reduced.rows.columns[term]] -= weight * reduced.rows.coefficients[term];
            }
        }
    }
    Some(weights)
}

fn reached_target(value: f64, options: OptimizeOptions) -> bool {
    options
        .target_lower
        .is_some_and(|target| value > target + 1e-8 * (1.0 + target.abs()))
}

struct NumericalCandidate {
    weights: Vec<f64>,
    value: f64,
    iterations: usize,
}

fn optimize_objective(
    reduced: &Reduced,
    objective: usize,
    options: OptimizeOptions,
    deadline: Instant,
) -> Option<NumericalCandidate> {
    let mut weights = backward_seed(reduced, objective, deadline, false)?;
    let mut residual = zeroes(reduced.free.len())?;
    let mut best_value = dual_value(reduced, objective, &weights, &mut residual)?;
    let forced = backward_seed(reduced, objective, deadline, true)?;
    let mut trial_residual = zeroes(reduced.free.len())?;
    let forced_value = dual_value(reduced, objective, &forced, &mut trial_residual)?;
    if forced_value > best_value {
        weights = forced;
        residual = trial_residual;
        best_value = forced_value;
    }
    let mut best_weights = weights.clone();
    let mut first_moment = zeroes(weights.len())?;
    let mut second_moment = zeroes(weights.len())?;
    let mut derivative = zeroes(residual.len())?;
    let magnitude = reduced.objectives[objective]
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    let learning = options.learning_rate * magnitude.max(1.0);
    if !learning.is_finite() {
        return None;
    }
    let mut anneal = 1.0;
    let mut beta1_power = 1.0;
    let mut beta2_power = 1.0;
    let mut iterations = 0;
    while iterations < options.steps
        && !expired(deadline)
        && !reached_target(best_value, options)
        && !weights.is_empty()
    {
        let smoothing = (magnitude * 1e-3 * anneal).max(1e-10);
        for (&r, d) in residual.iter().zip(&mut derivative) {
            *d = r / r.hypot(smoothing);
        }
        beta1_power *= 0.9;
        beta2_power *= 0.999;
        anneal *= options.anneal;
        let mut complete = true;
        for row in 0..weights.len() {
            if row % 256 == 0 && expired(deadline) {
                complete = false;
                break;
            }
            let mut gradient = reduced.rhs[row];
            for term in reduced.rows.terms(row) {
                gradient +=
                    reduced.rows.coefficients[term] * derivative[reduced.rows.columns[term]];
            }
            first_moment[row] = 0.9 * first_moment[row] + 0.1 * gradient;
            second_moment[row] = 0.999 * second_moment[row] + 0.001 * gradient * gradient;
            let direction = (first_moment[row] / (1.0 - beta1_power))
                / ((second_moment[row] / (1.0 - beta2_power)).sqrt() + 1e-12);
            let weight = (weights[row] + learning * anneal * direction).max(0.0);
            if !weight.is_finite() || weight / reduced.scales[row] > MAX_MULTIPLIER {
                complete = false;
                break;
            }
            weights[row] = weight;
        }
        if !complete {
            break;
        }
        iterations += 1;
        let Some(value) = dual_value(reduced, objective, &weights, &mut residual) else {
            break;
        };
        if value > best_value {
            best_value = value;
            best_weights.copy_from_slice(&weights);
        }
    }
    Some(NumericalCandidate {
        weights: best_weights,
        value: best_value,
        iterations,
    })
}

/// Lift reduced multipliers to the ORIGINAL authored equality rows.
fn recover(
    model: &Model,
    reduced: &Reduced,
    objective: usize,
    numerical: NumericalCandidate,
) -> Option<Candidate> {
    let output = objective / 2;
    let sign = if objective.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    let mut residual = zeroes(model.bounds.len())?;
    let (f, g) = model.outputs[output];
    residual[f] += sign;
    residual[g] -= sign;
    let mut inequality = numerical.weights;
    for ((weight, &scale), row) in inequality
        .iter_mut()
        .zip(&reduced.scales)
        .zip(&model.inequalities)
    {
        *weight /= scale;
        if !weight.is_finite() || *weight < 0.0 || *weight > MAX_MULTIPLIER {
            return None;
        }
        for &(column, coefficient) in &row.terms {
            residual[column] -= *weight * coefficient;
        }
    }
    let mut equality = zeroes(model.equalities.len())?;
    for &row_index in reduced.ordered_equalities.iter().rev() {
        let weight = residual[model.pivots[row_index]];
        if !weight.is_finite() || weight.abs() > MAX_MULTIPLIER {
            return None;
        }
        equality[row_index] = weight;
        for &(column, coefficient) in &model.equalities[row_index].terms {
            residual[column] -= weight * coefficient;
        }
    }
    if !finite(&residual) {
        return None;
    }
    Some(Candidate {
        output,
        sign,
        dual: DualMultipliers {
            equality_multipliers: equality,
            inequality_multipliers: inequality,
        },
        proposal_lower: numerical.value,
        iterations: numerical.iterations,
    })
}

/// Numerically propose both signs for every source-authenticated output.
/// Parallelism is across objectives; each objective has independent Adam
/// state, deterministic arithmetic, and the same shared absolute deadline.
/// Deadline expiry during reduction returns no candidate. Expiry after
/// reduction retains the best completed iterate (or its initial seed).
pub(super) fn optimize(
    model: &Model,
    options: OptimizeOptions,
    deadline: Instant,
) -> Option<OptimizationResult> {
    if !options.valid() {
        return None;
    }
    let reduced = reduce(model, deadline)?;
    let objectives = reduced.objectives.len();
    let results = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(objectives)
            .collect::<Vec<Option<Candidate>>>(),
    );
    let next = AtomicUsize::new(0);
    // OS thread creation failure returns no proposal, never a proof. All
    // successfully launched workers are joined by the scope before return.
    std::thread::scope(|scope| -> Option<()> {
        let mut handles = Vec::new();
        let mut okay = true;
        for _ in 0..options.workers.min(objectives) {
            match std::thread::Builder::new()
                .name("ny-paired-lp-proposal".into())
                .spawn_scoped(scope, || loop {
                    let objective = next.fetch_add(1, Ordering::Relaxed);
                    if objective >= objectives {
                        break;
                    }
                    let candidate = optimize_objective(&reduced, objective, options, deadline)
                        .and_then(|numerical| recover(model, &reduced, objective, numerical));
                    let Ok(mut results) = results.lock() else {
                        break;
                    };
                    results[objective] = candidate;
                }) {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    okay = false;
                    break;
                }
            }
        }
        for handle in handles {
            if handle.join().is_err() {
                okay = false;
            }
        }
        okay.then_some(())
    })?;
    let candidates = results
        .into_inner()
        .ok()?
        .into_iter()
        .collect::<Option<Vec<_>>>()?;
    Some(OptimizationResult {
        candidates,
        free_variables: reduced.free.len(),
        reduced_nonzeros: reduced.rows.columns.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paired_difference::{BoxDomain, PairedChain, Stage};
    use std::time::Duration;

    fn budget() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    fn options(steps: usize) -> OptimizeOptions {
        OptimizeOptions {
            steps,
            workers: 1,
            ..OptimizeOptions::default()
        }
    }

    fn replay(model: &Model, candidate: &Candidate) -> f64 {
        let mut objective = vec![0.0; model.bounds.len()];
        let (f, g) = model.outputs[candidate.output];
        objective[f] += candidate.sign;
        objective[g] -= candidate.sign;
        super::super::lp_certificate::certified_lower(
            &model.bounds,
            &objective,
            &model.equalities,
            &model.inequalities,
            &candidate.dual,
        )
        .expect("native proposal must replay through the independent checker")
    }

    #[test]
    fn affine_cancellation_lifts_to_original_rows_and_checker() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![2.0],
                wg: vec![2.0],
                bf: vec![3.0],
                bg: vec![4.0],
            }],
        };
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let model = super::super::lp_model::build(&chain, &domain).unwrap();
        let result = optimize(&model, options(0), budget()).unwrap();
        assert_eq!(result.candidates.len(), 2);
        for (candidate, exact) in result.candidates.iter().zip([-1.0, 1.0]) {
            assert!((candidate.proposal_lower - exact).abs() < 1e-12);
            let checked = replay(&model, candidate);
            assert!(checked <= exact && checked > exact - 1e-10);
        }
    }

    #[test]
    fn coupled_relu_seed_cancels_identical_towers_and_replays() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![1.0],
                    wg: vec![1.0],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
                Stage::Relu { width: 1 },
            ],
        };
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let model = super::super::lp_model::build(&chain, &domain).unwrap();
        let result = optimize(&model, options(20), budget()).unwrap();
        for candidate in &result.candidates {
            assert!(candidate.proposal_lower.abs() < 1e-10);
            assert!(candidate
                .dual
                .inequality_multipliers
                .iter()
                .all(|x| *x >= 0.0));
            let checked = replay(&model, candidate);
            assert!(checked <= 0.0 && checked > -1e-8);
        }
    }

    #[test]
    fn reverse_substitution_crosses_a_loose_intermediate_box_and_replays() {
        // x >= y - 2 and y >= z + 3 imply x >= z + 1 >= 0.
        // Substituting only the first row worsens the independent box bound.
        let reduced = Reduced {
            free: vec![0, 1, 2],
            ordered_equalities: vec![],
            rows: SparseRows {
                offsets: vec![0, 2, 4],
                columns: vec![1, 2, 0, 1],
                coefficients: vec![-10.0, 1.0, -1.0, 10.0],
                ending_at: vec![vec![], vec![1], vec![0]],
            },
            rhs: vec![-2.0, 3.0],
            scales: vec![1.0; 2],
            objectives: vec![vec![0.0, 0.0, 1.0]],
            constants: vec![0.0],
        };
        let mut residual = vec![0.0; 3];
        let local = backward_seed(&reduced, 0, budget(), false).unwrap();
        let forced = backward_seed(&reduced, 0, budget(), true).unwrap();
        assert_eq!(dual_value(&reduced, 0, &local, &mut residual), Some(-1.0));
        assert_eq!(dual_value(&reduced, 0, &forced, &mut residual), Some(0.0));
        let rows = vec![
            LinearRow {
                terms: vec![(1, -1.0), (2, 1.0)],
                rhs: -2.0,
            },
            LinearRow {
                terms: vec![(0, -1.0), (1, 1.0)],
                rhs: 3.0,
            },
        ];
        // Original rows before normalizing y to its [-10,10] box. The residual certificate
        // cancels y, so only the z lower bound remains relevant.
        let checked = super::super::lp_certificate::certified_lower(
            &[(-1.0, 1.0), (-10.0, 10.0), (-1.0, 1.0)],
            &[0.0, 0.0, 1.0],
            &[],
            &rows,
            &DualMultipliers {
                equality_multipliers: vec![],
                inequality_multipliers: forced,
            },
        )
        .unwrap();
        assert!(checked <= 0.0 && checked > -1e-12);
    }

    #[test]
    fn invalid_controls_and_expired_reduction_return_no_proposal() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![1.0],
                wg: vec![1.0],
                bf: vec![0.0],
                bg: vec![0.0],
            }],
        };
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let model = super::super::lp_model::build(&chain, &domain).unwrap();
        assert!(optimize(&model, options(0), Instant::now()).is_none());
        for invalid in [
            OptimizeOptions {
                workers: 0,
                ..options(0)
            },
            OptimizeOptions {
                workers: 17,
                ..options(0)
            },
            OptimizeOptions {
                learning_rate: f64::NAN,
                ..options(0)
            },
            OptimizeOptions {
                anneal: 0.0,
                ..options(0)
            },
            OptimizeOptions {
                target_lower: Some(f64::INFINITY),
                ..options(0)
            },
        ] {
            assert!(optimize(&model, invalid, budget()).is_none());
        }
    }
}
