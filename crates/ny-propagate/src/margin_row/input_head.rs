// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! A separately owned input-box producer for a static FLOAT32 dense head.
//! No TwinNet, RootGates, YPack, artificial layer or additional class exists
//! here. Advice proposes slopes or row duals; all arithmetic is replayed locally.
//! Certificate authority is the mathematical ONNX composition over exact
//! stored FLOAT32 constants. Additional balls also enclose RN/gradual IEEE
//! FLOAT32 paths; this does not qualify an external TF32/FTZ/GPU executor.

use super::advice::AdviceBudget;
use super::*;
use crate::certify::{authorize_unsat, TokenScope, UnsatToken};
use crate::guards::containment::{run_containment, CertifiedFrame, ContainmentAudit};
use ndarray::Array2;
use ny_core::head_functional_advice::{
    HeadFunctionalLpRequest, HeadFunctionalLpRow, FUNCTIONAL_PROTOCOL_VERSION,
};
use ny_core::head_phase_advice::HeadPhaseAdvice;
use std::sync::Arc;

const MAX_INPUTS: usize = 128;
const MAX_HEADS: usize = 512;
const F32_U: f64 = 1.0 / 16_777_216.0;
const F32_ETA: f64 = f32::from_bits(1) as f64;
const F32_LIMIT: f64 = f32::MAX as f64 / 4.0;

/// Immutable authored parameters. Matrices have ONNX's constant-right
/// row-major layout [input, hidden] and [hidden, output]. Only exact FLOAT32
/// storage is admitted; this constructor does not accept folded coefficients.
/// Numeric certificates cannot be supplied, serialized or mutated by callers.
#[derive(Clone, Debug)]
pub struct InputHeadSpec {
    inputs: usize,
    heads: usize,
    first: Vec<f32>,
    first_bias: Vec<f32>,
    final_weight: Vec<f32>,
    final_bias: Vec<f32>,
    authored_digest: [u8; 32],
}

impl InputHeadSpec {
    pub fn new(
        inputs: usize,
        heads: usize,
        first: Vec<f32>,
        first_bias: Vec<f32>,
        final_weight: Vec<f32>,
        final_bias: Vec<f32>,
        authored_digest: [u8; 32],
    ) -> Option<Self> {
        if !(1..=MAX_INPUTS).contains(&inputs)
            || !(1..=MAX_HEADS).contains(&heads)
            || first.len() != inputs.checked_mul(heads)?
            || first_bias.len() != heads
            || final_weight.len() != heads.checked_mul(2)?
            || final_bias.len() != 2
            || first
                .iter()
                .chain(&first_bias)
                .chain(&final_weight)
                .chain(&final_bias)
                .any(|x| !x.is_finite())
        {
            return None;
        }
        Some(Self {
            inputs,
            heads,
            first,
            first_bias,
            final_weight,
            final_bias,
            authored_digest,
        })
    }
    pub fn inputs(&self) -> usize {
        self.inputs
    }
    pub fn heads(&self) -> usize {
        self.heads
    }
    pub fn authored_digest(&self) -> [u8; 32] {
        self.authored_digest
    }
}

#[derive(Debug, Default)]
pub struct InputHeadReport {
    pub lower: Option<f64>,
    pub advice_calls: usize,
    pub replayed_candidates: usize,
    pub elapsed_seconds: f64,
    pub triangle_advice_calls: usize,
    pub triangle_replayed_candidates: usize,
    pub triangle_rows: usize,
    pub triangle_variables: usize,
    pub triangle_cuts: usize,
    pub triangle_classes_improved: usize,
    pub triangle_before_lower: Option<f64>,
    pub triangle_after_lower: Option<f64>,
    pub triangle_negative_groups: usize,
    pub triangle_negative_group_masks: usize,
    pub triangle_large_negative_groups: usize,
    pub triangle_large_negative_group_masks: usize,
    pub phase_cover_selected_heads: usize,
    pub phase_cover_leaves: usize,
    pub phase_cover_advice_calls: usize,
    pub phase_cover_replayed_leaves: usize,
    pub phase_cover_native_replayed_leaves: usize,
    pub phase_cover_ay_replayed_leaves: usize,
    pub phase_cover_before_lower: Option<f64>,
    pub phase_cover_after_lower: Option<f64>,
    pub phase_cover_elapsed_seconds: f64,
    pub directional_candidates: usize,
    pub directional_certified: usize,
    pub directional_cuts: usize,
    pub directional_masks: usize,
    pub directional_work_cells: usize,
    pub directional_advice_calls: usize,
    pub directional_native_replayed: usize,
    pub directional_ay_replayed: usize,
    pub directional_before_lower: Option<f64>,
    pub directional_after_lower: Option<f64>,
    pub directional_elapsed_seconds: f64,
    pub directional_rounds: usize,
    pub directional_attempted: usize,
    pub directional_affine_cuts: usize,
}

pub enum InputHeadOutcome {
    Unsat(InputHeadReport, UnsatToken),
    Unknown(InputHeadReport),
}

/// This fresh Arc generation owns the original graph constants, entire input
/// box and original singleton class mapping. A candidate never brings its
/// own box, Q, arithmetic charge, source or lower bound.
#[derive(Clone)]
struct Source {
    spec: Arc<InputHeadSpec>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    target: usize,
    adversary: usize,
    y: Vec<I>,
    first_error: Vec<f64>,
    q: Vec<I>,
    bias: I,
    output_error: f64,
}

fn max_abs(x: I) -> f64 {
    x.lo.abs().max(x.hi.abs())
}

/// Standard RN FLOAT32 dot+Add envelope, permitting product/reduction FMA
/// or reassociation with at most 2*n+1 rounding sites on any term path.
/// Each site contributes one eta (including products rounded to zero).
/// All absolute partial sums stay below MAX/4: overflow is refused up front.
fn dot_error(absolute_sum: f64, terms: usize) -> Option<f64> {
    if !good(absolute_sum) || !(0.0..=F32_LIMIT).contains(&absolute_sum) {
        return None;
    }
    let k = terms.checked_mul(2)?.checked_add(1)? as f64;
    let ku = k * F32_U; // exact at the admitted counts
    let denominator = (1.0 - ku).next_down();
    if denominator <= 0.75 {
        return None;
    }
    let gamma = (ku / denominator).next_up();
    let absolute = ((k * F32_ETA) / denominator).next_up();
    Some(I::p(gamma).mul(I::p(absolute_sum))?.add(I::p(absolute))?.hi)
}

impl Source {
    fn build(
        spec: &InputHeadSpec,
        lo: &[f64],
        hi: &[f64],
        target: usize,
        adversary: usize,
        deadline: Instant,
    ) -> Option<Self> {
        if !live(deadline)
            || !fp_qualified()
            || lo.len() != spec.inputs
            || hi.len() != lo.len()
            || target >= 2
            || adversary >= 2
            || target == adversary
        {
            return None;
        }
        let root: Vec<I> = lo.iter().zip(hi).map(|(&lo, &hi)| I { lo, hi }).collect();
        if root.iter().any(|x| !x.valid() || max_abs(*x) > F32_LIMIT) {
            return None;
        }
        let input_error: Vec<f64> = root
            .iter()
            .map(|&x| {
                I::p(F32_U)
                    .mul(I::p(max_abs(x)))?
                    .add(I::p(F32_ETA))
                    .map(|e| e.hi)
            })
            .collect::<Option<_>>()?;
        let mut y = Vec::with_capacity(spec.heads);
        let mut first_error = Vec::with_capacity(spec.heads);
        for j in 0..spec.heads {
            if !live(deadline) {
                return None;
            }
            let mut interval = I::p(f64::from(spec.first_bias[j]));
            let mut absolute = I::p(f64::from(spec.first_bias[j]).abs());
            let mut input_charge = I::p(0.0);
            for k in 0..spec.inputs {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                let w = f64::from(spec.first[k * spec.heads + j]);
                interval = interval.add(I::p(w).mul(root[k])?)?;
                input_charge = input_charge.add(I::p(w.abs()).mul(I::p(input_error[k]))?)?;
                absolute = absolute
                    .add(I::p(w.abs()).mul(I::p(max_abs(root[k])).add(I::p(input_error[k]))?)?)?;
            }
            let error = input_charge
                .add(I::p(dot_error(absolute.hi, spec.inputs)?))?
                .hi;
            y.push(interval.add(I {
                lo: -error,
                hi: error,
            })?);
            first_error.push(error);
        }
        let mut q = Vec::with_capacity(spec.heads);
        let mut final_errors = [0.0; 2];
        for (output, error) in final_errors.iter_mut().enumerate() {
            let mut absolute = I::p(f64::from(spec.final_bias[output]).abs());
            for j in 0..spec.heads {
                if j & 63 == 0 && !live(deadline) {
                    return None;
                }
                absolute = absolute.add(
                    I::p(f64::from(spec.final_weight[j * 2 + output]).abs())
                        .mul(I::p(y[j].hi.max(0.0)))?,
                )?;
            }
            *error = dot_error(absolute.hi, spec.heads)?;
        }
        for j in 0..spec.heads {
            if j & 63 == 0 && !live(deadline) {
                return None;
            }
            q.push(
                I::p(f64::from(spec.final_weight[j * 2 + target]))
                    .sub(I::p(f64::from(spec.final_weight[j * 2 + adversary])))?,
            );
        }
        let bias = I::p(f64::from(spec.final_bias[target]))
            .sub(I::p(f64::from(spec.final_bias[adversary])))?;
        let output_error = I::p(final_errors[target])
            .add(I::p(final_errors[adversary]))?
            .hi;
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        Some(Self {
            spec: Arc::new(spec.clone()),
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            target,
            adversary,
            y,
            first_error,
            q,
            bias,
            output_error,
        })
    }

    fn baseline(&self, deadline: Instant) -> Option<f64> {
        let mut b = self.bias;
        for (&q, &y) in self.q.iter().zip(&self.y) {
            if !live(deadline) {
                return None;
            }
            b = b.add(q.mul(I {
                lo: y.lo.max(0.0),
                hi: y.hi.max(0.0),
            })?)?;
        }
        Some(b.sub(I::p(self.output_error))?.lo)
    }

    fn slopes(&self) -> Option<Vec<f64>> {
        self.q
            .iter()
            .zip(&self.y)
            .map(|(&q, &y)| {
                let q = 0.5 * q.lo + 0.5 * q.hi;
                let alpha = if y.hi <= 0.0 {
                    0.0
                } else if y.lo >= 0.0 {
                    1.0
                } else if q < 0.0 {
                    y.hi / (y.hi - y.lo)
                } else {
                    f64::from(u8::from(y.hi >= -y.lo))
                };
                let v = q * alpha;
                good(v).then_some(v)
            })
            .collect()
    }

    /// Full source functional, including every head/feature, both output
    /// execution charges and amplified first-layer errors. Directed products
    /// charge binary64 underflow before cancellation or amplification.
    fn replay(&self, slopes: &[f64], deadline: Instant) -> Option<f64> {
        self.replay_retained(slopes, deadline)
            .map(|(bound, _)| bound)
    }

    fn replay_retained(
        &self,
        slopes: &[f64],
        deadline: Instant,
    ) -> Option<(f64, bridge::ResidualForm)> {
        if slopes.len() != self.spec.heads || !fp_qualified() {
            return None;
        }
        let mut constant = self.bias.sub(I::p(self.output_error))?;
        let mut coefficients = vec![I::p(0.0); self.spec.inputs];
        for (j, &v) in slopes.iter().enumerate() {
            if !live(deadline) || !good(v) {
                return None;
            }
            constant = constant
                .add(I::p(scalar_intercept(self.q[j], v, self.y[j])?))?
                .add(I::p(v).mul(I::p(f64::from(self.spec.first_bias[j])))?)?
                .sub(I::p(v.abs()).mul(I::p(self.first_error[j]))?)?;
            for (k, a) in coefficients.iter_mut().enumerate() {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                *a =
                    a.add(I::p(v).mul(I::p(f64::from(self.spec.first[k * self.spec.heads + j])))?)?;
            }
        }
        let form = bridge::ResidualForm {
            generation: Arc::clone(&self.spec),
            constant,
            input: coefficients.clone(),
            relu: vec![I::p(0.0); self.spec.heads],
        };
        for (k, a) in coefficients.into_iter().enumerate() {
            if !live(deadline) {
                return None;
            }
            constant = constant.add(a.mul(I {
                lo: self.lo[k],
                hi: self.hi[k],
            })?)?;
        }
        (live(deadline) && fp_qualified()).then_some((constant.lo, form))
    }

    /// Complete dense geometry is small (<=128 features). It remains advice:
    /// the epigraph/status/objective never supplies a certified endpoint.
    fn export(&self, deadline: Instant) -> Option<HeadFunctionalLpRequest> {
        let slopes = self.slopes()?;
        let mut r = HeadFunctionalLpRequest {
            protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
            feature_count: self.spec.inputs,
            bounds: self
                .lo
                .iter()
                .zip(&self.hi)
                .map(|(&l, &h)| [l, h])
                .collect(),
            objective: vec![0.0; self.spec.inputs],
            rows: vec![],
            slope_rows: vec![None; self.spec.heads],
            phase_rows: vec![],
        };
        for j in 0..self.spec.heads {
            if !live(deadline) {
                return None;
            }
            let q = 0.5 * self.q[j].lo + 0.5 * self.q[j].hi;
            if q > 0.0 {
                let epigraph = r.bounds.len();
                r.bounds.push([0.0, self.y[j].hi.max(0.0)]);
                r.objective.push(q);
                r.rows.push(HeadFunctionalLpRow {
                    coefficients: vec![(epigraph, 1.0)],
                    lower: 0.0,
                });
                let mut coefficients = Vec::with_capacity(self.spec.inputs + 1);
                for k in 0..self.spec.inputs {
                    let w = f64::from(self.spec.first[k * self.spec.heads + j]);
                    if w != 0.0 {
                        coefficients.push((k, -w));
                    }
                }
                coefficients.push((epigraph, 1.0));
                r.slope_rows[j] = Some(r.rows.len());
                r.rows.push(HeadFunctionalLpRow {
                    coefficients,
                    lower: f64::from(self.spec.first_bias[j]),
                });
            } else {
                for (k, a) in r.objective[..self.spec.inputs].iter_mut().enumerate() {
                    *a += slopes[j] * f64::from(self.spec.first[k * self.spec.heads + j]);
                }
            }
        }
        (live(deadline) && r.validate()).then_some(r)
    }
}

struct Frame<'a> {
    source: &'a Source,
    lower: f64,
    deadline: Instant,
}
impl CertifiedFrame for Frame<'_> {
    fn input_box(&self) -> (&[f64], &[f64]) {
        (&self.source.lo, &self.source.hi)
    }
    fn point_cost_secs(&self) -> f64 {
        (self.source.spec.inputs * self.source.spec.heads * 2) as f64 / 5e8
    }
    fn point_bytes(&self) -> u64 {
        ((self.source.spec.inputs + self.source.spec.heads + 2) * 8) as u64
    }
    fn audit(&self, points: &Array2<f64>, audit: &mut ContainmentAudit) -> Result<(), String> {
        let s = self.source;
        let spec = &s.spec;
        if points.nrows() != spec.inputs {
            return Err("dense G2 input dimension mismatch".into());
        }
        for sample in 0..points.ncols() {
            if !live(self.deadline) || !fp_qualified() {
                return Err("dense G2 deadline/FP refusal".into());
            }
            let mut h = Vec::with_capacity(spec.heads);
            for j in 0..spec.heads {
                let mut a = 0.0f32;
                for k in 0..spec.inputs {
                    if k & 63 == 0 && !live(self.deadline) {
                        return Err("dense G2 expired".into());
                    }
                    a += spec.first[k * spec.heads + j] * (points[[k, sample]] as f32);
                }
                a += spec.first_bias[j];
                if !a.is_finite() {
                    return Err("dense G2 nonfinite preactivation".into());
                }
                audit.check_box(
                    "input_head_y",
                    j,
                    sample,
                    s.y[j].lo,
                    s.y[j].hi,
                    f64::from(a),
                );
                h.push(a.max(0.0));
            }
            let mut outputs = [0.0f32; 2];
            for (o, a) in outputs.iter_mut().enumerate() {
                for j in 0..spec.heads {
                    if j & 63 == 0 && !live(self.deadline) {
                        return Err("dense G2 expired".into());
                    }
                    *a += spec.final_weight[j * 2 + o] * h[j];
                }
                *a += spec.final_bias[o];
                if !a.is_finite() {
                    return Err("dense G2 nonfinite output".into());
                }
            }
            let margin = f64::from(outputs[s.target]) - f64::from(outputs[s.adversary]);
            audit.check_lower("input_head_margin", s.adversary, sample, self.lower, margin);
            audit.check_positive(
                "input_head_margin",
                s.adversary,
                sample,
                margin,
                f64::from(outputs[0]).abs().max(f64::from(outputs[1]).abs()),
            );
        }
        Ok(())
    }
}

/// Certify the ONE original margin on the entire provided box. The caller
/// must bind this source to the original graph and authenticated property;
/// CLI publication additionally ratifies the token through ordinary G3.
pub fn verify_input_head(
    spec: &InputHeadSpec,
    lo: &[f64],
    hi: &[f64],
    target: usize,
    adversary: usize,
    policy: HeadGroupPolicy,
    deadline: Instant,
) -> InputHeadOutcome {
    verify_with_provider(
        spec,
        lo,
        hi,
        target,
        adversary,
        policy,
        deadline,
        advice::registered_provider,
    )
}

/// Preserve the immediate original-box route, then retain only completed
/// source-owned functionals for the actual evaluation root. Proposal
/// preparation, two provider calls and re-pricing share the original250ms
/// AdviceBudget; the original native baseline keeps its existing outer slice.
/// The caller authenticates original graph/property and the actual reduction.
pub fn verify_input_head_retained(
    spec: &InputHeadSpec,
    lo: &[f64],
    hi: &[f64],
    target: usize,
    adversary: usize,
    policy: HeadGroupPolicy,
    authority: Instant,
    evaluation_lo: &[f64],
    evaluation_hi: &[f64],
    preparation_deadline: Instant,
) -> (InputHeadOutcome, Option<Arc<InputHeadContinuation>>) {
    verify_with_provider_retained(
        spec,
        lo,
        hi,
        target,
        adversary,
        policy,
        authority,
        advice::registered_provider,
        Some((evaluation_lo, evaluation_hi, preparation_deadline)),
    )
}

/// Supply finite proposal hints to one test invocation without modifying the
/// process-global advice factory. All rows, replay, retention and authority
/// still come from the ordinary source-owned producer below.
#[cfg(test)]
pub(crate) fn test_verify_input_head_retained_with_provider<F>(
    spec: &InputHeadSpec,
    lo: &[f64],
    hi: &[f64],
    target: usize,
    adversary: usize,
    policy: HeadGroupPolicy,
    authority: Instant,
    evaluation_lo: &[f64],
    evaluation_hi: &[f64],
    preparation_deadline: Instant,
    provider: F,
) -> (InputHeadOutcome, Option<Arc<InputHeadContinuation>>)
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    verify_with_provider_retained(
        spec,
        lo,
        hi,
        target,
        adversary,
        policy,
        authority,
        provider,
        Some((evaluation_lo, evaluation_hi, preparation_deadline)),
    )
}

fn verify_with_provider<F>(
    spec: &InputHeadSpec,
    lo: &[f64],
    hi: &[f64],
    target: usize,
    adversary: usize,
    policy: HeadGroupPolicy,
    deadline: Instant,
    provider: F,
) -> InputHeadOutcome
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    verify_with_provider_retained(
        spec, lo, hi, target, adversary, policy, deadline, provider, None,
    )
    .0
}

fn verify_with_provider_retained<F>(
    spec: &InputHeadSpec,
    lo: &[f64],
    hi: &[f64],
    target: usize,
    adversary: usize,
    policy: HeadGroupPolicy,
    deadline: Instant,
    provider: F,
    evaluation: Option<(&[f64], &[f64], Instant)>,
) -> (InputHeadOutcome, Option<Arc<InputHeadContinuation>>)
where
    F: Fn() -> Option<Arc<dyn HeadPhaseAdvice>>,
{
    let start = Instant::now();
    let mut report = InputHeadReport::default();
    let mut continuation = None;
    let mut completed = Vec::new();
    let mut recipes = Vec::new();
    let result = (|| {
        let slice = start
            .checked_add(Duration::from_secs_f64(policy.max_seconds()))?
            .min(deadline)
            .min(evaluation.map_or(deadline, |(_, _, prep)| prep));
        let source = Source::build(spec, lo, hi, target, adversary, slice)?;
        let mut best = source.baseline(slice)?; // complete original singleton cover before any provider
        if let Some(slopes) = source.slopes() {
            let replayed = if evaluation.is_some() {
                source
                    .replay_retained(&slopes, slice)
                    .map(|(bound, form)| (bound, Some(form)))
            } else {
                source.replay(&slopes, slice).map(|bound| (bound, None))
            };
            if let Some((bound, form)) = replayed {
                best = best.max(bound);
                report.replayed_candidates += 1;
                if let Some(form) = form {
                    completed.push(form);
                    recipes.push(bridge::Recipe::Slopes(slopes));
                }
            }
        }
        let mut budget = AdviceBudget::new();
        let mut phase_prefix = None;
        if best <= 0.0 {
            let prep = Instant::now();
            let proposed = (|| {
                let prep_deadline = budget.call_deadline(prep, slice)?;
                let provider = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&provider))
                    .ok()
                    .flatten()?;
                let supported = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    provider.supports_functional_advice()
                }))
                .ok()?;
                if !supported || !live(prep_deadline) {
                    return None;
                }
                let request = source.export(prep_deadline)?;
                let call_deadline = budget
                    .call_deadline(Instant::now(), slice)?
                    .min(prep_deadline);
                budget.calls += 1;
                let hint = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    provider.propose_functional(&request, call_deadline)
                }))
                .ok()
                .flatten()?;
                (live(call_deadline) && hint.validate(&request)).then_some((request, hint))
            })();
            budget.charge_elapsed(prep.elapsed()); // includes every unsupported/panic/early-return path
            report.advice_calls = budget.calls;
            let replay_begin = Instant::now();
            if let Some((request, hint)) = proposed {
                let replay_deadline = budget.replay_deadline(replay_begin, slice);
                for sign in [1.0, -1.0] {
                    let Some(replay_deadline) = replay_deadline else {
                        break;
                    };
                    let mut slopes = source.slopes()?;
                    for (j, row) in request.slope_rows.iter().enumerate() {
                        if let Some(row) = row {
                            slopes[j] = (sign * hint.row_duals[*row]).max(0.0);
                        }
                    }
                    let replayed = if evaluation.is_some() {
                        source
                            .replay_retained(&slopes, replay_deadline)
                            .map(|(bound, form)| (bound, Some(form)))
                    } else {
                        source
                            .replay(&slopes, replay_deadline)
                            .map(|bound| (bound, None))
                    };
                    if let Some((bound, form)) = replayed {
                        best = best.max(bound);
                        report.replayed_candidates += 1;
                        if let Some(form) = form {
                            completed.push(form);
                            recipes.push(bridge::Recipe::Slopes(slopes));
                        }
                    }
                }
            }
            budget.charge_elapsed(replay_begin.elapsed());
        }
        // Keep the completed original scalar/slope floor and first advice
        // opportunity. This second model shares the SAME2call/250ms ledger.
        // Its actual input+ReLU roles use V2 honestly; all signed-Q rows are
        // replayed independently rather than interpreted as positive slopes.
        if best <= 0.0 {
            let before_triangle = best;
            report.triangle_before_lower = Some(before_triangle);
            let before_forms = completed.len();
            let before_recipes = recipes.len();
            let calls_before = budget.calls;
            let prep = Instant::now();
            let proposed = (|| {
                let prep_deadline = budget.call_deadline(prep, slice)?;
                let provider = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&provider))
                    .ok()
                    .flatten()?;
                let supported = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    provider.supports_functional_advice()
                }))
                .ok()?;
                if !supported || !live(prep_deadline) {
                    return None;
                }
                let model = triangle::Model::build(&source, prep_deadline)?;
                let call_deadline = budget
                    .call_deadline(Instant::now(), slice)?
                    .min(prep_deadline);
                budget.calls += 1;
                let hint = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    provider.propose_functional(model.request(), call_deadline)
                }))
                .ok()
                .flatten()
                .filter(|hint| live(call_deadline) && hint.validate(model.request()));
                // The complete owned rows predate and do not trust advice.
                // Preserve them for an armed cover even if AY declines.
                if hint.is_none() && policy.phase_cover().is_none() && !policy.directional_cuts() {
                    return None;
                }
                Some((model, hint))
            })();
            budget.charge_elapsed(prep.elapsed());
            report.triangle_advice_calls = budget.calls - calls_before;
            report.advice_calls = budget.calls;
            let replay_begin = Instant::now();
            if let Some((model, hint)) = proposed {
                if let Some(hint) = hint {
                    let replay_deadline = budget.replay_deadline(replay_begin, slice);
                    report.triangle_rows = model.request().rows.len();
                    report.triangle_variables = model.request().bounds.len();
                    report.triangle_cuts = model.cuts();
                    report.triangle_negative_groups = model.negative_groups();
                    report.triangle_negative_group_masks = model.negative_group_masks();
                    report.triangle_large_negative_groups = model.large_negative_groups();
                    report.triangle_large_negative_group_masks = model.large_negative_group_masks();
                    for sign in [1.0, -1.0] {
                        let Some(replay_deadline) = replay_deadline else {
                            break;
                        };
                        if let Some((bound, form, recipe)) = model.replay_recorded(
                            &source,
                            &hint,
                            sign,
                            replay_deadline,
                            evaluation.is_some(),
                        ) {
                            best = best.max(bound);
                            report.triangle_replayed_candidates += 1;
                            report.replayed_candidates += 1;
                            if let (Some(form), Some(recipe)) = (form, recipe) {
                                completed.push(form);
                                recipes.push(recipe);
                            }
                        }
                    }
                }
                if policy.phase_cover().is_some() || policy.directional_cuts() {
                    phase_prefix = Some(model.into_phase_prefix());
                }
            }
            budget.charge_elapsed(replay_begin.elapsed());
            if !live(slice) || !live(deadline) || !fp_qualified() {
                completed.truncate(before_forms);
                recipes.truncate(before_recipes);
            }
            best = commit_triangle_best(before_triangle, best, slice, deadline);
            report.triangle_after_lower = Some(best);
            report.triangle_classes_improved = usize::from(best > before_triangle);
        }
        // One bounded root geometry stage, shared by directional construction,
        // the joint LP and any subsequent optional nominal-sign cover. The
        // earlier two-call/250ms legacy ledger is never reset or extended.
        let root_stage = Instant::now()
            .checked_add(Duration::from_millis(750))?
            .min(slice)
            .min(deadline);
        if policy.directional_cuts() {
            report.directional_before_lower = Some(best);
            if best <= 0.0 {
                // Predetermined remaining-time split: leave 2/5 to the later
                // cover, rather than renewing authority after each LP round.
                let directional_deadline = directional_stage_deadline(
                    Instant::now(),
                    root_stage,
                    policy.phase_cover().is_some(),
                );
                let (refined, prefix) = directional::refine_with_prefix(
                    &source,
                    best,
                    phase_prefix,
                    directional_deadline,
                    deadline,
                    &provider,
                );
                best = refined.lower;
                phase_prefix = prefix;
                report.directional_candidates = refined.candidates;
                report.directional_certified = refined.certified;
                report.directional_cuts = refined.cuts;
                report.directional_masks = refined.masks;
                report.directional_work_cells = refined.work_cells;
                report.directional_advice_calls = refined.calls;
                report.directional_native_replayed = refined.native_replayed;
                report.directional_ay_replayed = refined.ay_replayed;
                report.directional_elapsed_seconds = refined.elapsed_seconds;
                report.directional_rounds = refined.rounds.len();
                report.directional_attempted = refined.attempted;
                report.directional_affine_cuts = refined.affine_cuts;
                report.replayed_candidates += refined.native_replayed + refined.ay_replayed;
            }
            report.directional_after_lower = Some(best);
        }
        if let Some(phase_policy) = policy.phase_cover() {
            report.phase_cover_before_lower = Some(best);
            if best <= 0.0 {
                let refined = match phase_prefix {
                    Some(prefix) => phase_cover::refine_with_prefix(
                        &source,
                        best,
                        phase_policy,
                        Some(prefix),
                        root_stage,
                        deadline,
                        &provider,
                    ),
                    None => phase_cover::refine(
                        &source,
                        best,
                        phase_policy,
                        root_stage,
                        deadline,
                        &provider,
                    ),
                };
                best = refined.lower;
                report.phase_cover_selected_heads = refined.selected_heads;
                report.phase_cover_leaves = refined.leaves;
                report.phase_cover_advice_calls = refined.calls;
                report.phase_cover_replayed_leaves = refined.replayed;
                report.phase_cover_native_replayed_leaves = refined.native_replayed;
                report.phase_cover_ay_replayed_leaves = refined.ay_replayed;
                report.phase_cover_elapsed_seconds = refined.elapsed_seconds;
                report.replayed_candidates += refined.replayed;
            }
            report.phase_cover_after_lower = Some(best);
        }
        report.lower = Some(best);
        if best <= 0.0 {
            if let Some((evaluation_lo, evaluation_hi, _)) = evaluation {
                let prep = Instant::now();
                if let Some(end) = budget.replay_deadline(prep, slice) {
                    continuation = InputHeadContinuation::build(
                        &source,
                        &completed,
                        &recipes,
                        evaluation_lo,
                        evaluation_hi,
                        end,
                        deadline,
                    );
                }
                budget.charge_elapsed(prep.elapsed());
            }
        }
        let cover = crate::certify::margin_row::root_class(RoundMode::Outward, adversary, best)?;
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        let frame = Frame {
            source: &source,
            lower: best,
            deadline,
        };
        let allowance = deadline
            .saturating_duration_since(Instant::now())
            .mul_f64(0.02);
        let g2 = run_containment(
            &frame,
            ny_core::proof_profile::audit_rates().containment_points,
            Some(allowance),
        );
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        let token =
            authorize_unsat(TokenScope::Instance, &[adversary], &[cover], &g2.report()).ok()?;
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        eprintln!(
            "[cover-gate] authorized scope=instance classes=1 tree=none {}",
            g2.fragment()
        );
        Some(token)
    })();
    report.elapsed_seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "[input-head-triangle] calls={} replayed={} variables={} rows={} jensen_cuts={} classes_improved={}",
        report.triangle_advice_calls, report.triangle_replayed_candidates,
        report.triangle_variables, report.triangle_rows, report.triangle_cuts,
        report.triangle_classes_improved
    );
    eprintln!(
        "[input-head] status={} classes=1 calls={} replayed={} lower_bits={} elapsed_s={:.6}",
        if result.is_some() { "complete" } else { "open" },
        report.advice_calls,
        report.replayed_candidates,
        report
            .lower
            .map_or_else(|| "none".into(), |v| format!("{:016x}", v.to_bits())),
        report.elapsed_seconds
    );
    if let (Some(before), Some(after)) = (report.triangle_before_lower, report.triangle_after_lower)
    {
        eprintln!(
            "[input-head-triangle-delta] before_bits={:016x} after_bits={:016x}",
            before.to_bits(),
            after.to_bits()
        );
    }
    eprintln!(
        "[input-head-negative-group-replay] rows={} certified_masks={}",
        report.triangle_negative_groups, report.triangle_negative_group_masks
    );
    eprintln!(
        "[input-head-large-negative-group-replay] rows={} certified_masks={}",
        report.triangle_large_negative_groups, report.triangle_large_negative_group_masks
    );
    if policy.directional_cuts() {
        eprintln!(
            "[input-head-directional] candidates={} certified={} cuts={} algebraic_masks={} work_cells={} calls={} native_replayed={} ay_replayed={} before_bits={} after_bits={} elapsed_s={:.6}",
            report.directional_candidates, report.directional_certified,
            report.directional_cuts, report.directional_masks, report.directional_work_cells,
            report.directional_advice_calls, report.directional_native_replayed, report.directional_ay_replayed,
            report.directional_before_lower.map_or_else(|| "none".into(), |value| format!("{:016x}", value.to_bits())),
            report.directional_after_lower.map_or_else(|| "none".into(), |value| format!("{:016x}", value.to_bits())),
            report.directional_elapsed_seconds,
        );
    }
    if policy.directional_cuts() {
        eprintln!("[input-head-directional-portfolio] rounds={} attempted={} affine_cuts={} max_rounds=3 max_attempts=32 max_masks=8192 max_cells=2000000 cover_reserve_fifths={}",
            report.directional_rounds, report.directional_attempted, report.directional_affine_cuts,
            usize::from(policy.phase_cover().is_some()) * 2);
    }
    if policy.phase_cover().is_some() {
        eprintln!(
            "[input-head-phase-cover] calls={} leaves={} replayed={} native_replayed={} ay_replayed={} selected_heads={} before_bits={} after_bits={} elapsed_s={:.6}",
            report.phase_cover_advice_calls,
            report.phase_cover_leaves,
            report.phase_cover_replayed_leaves,
            report.phase_cover_native_replayed_leaves,
            report.phase_cover_ay_replayed_leaves,
            report.phase_cover_selected_heads,
            report.phase_cover_before_lower.map_or_else(|| "none".into(), |value| format!("{:016x}", value.to_bits())),
            report.phase_cover_after_lower.map_or_else(|| "none".into(), |value| format!("{:016x}", value.to_bits())),
            report.phase_cover_elapsed_seconds,
        );
    }
    let outcome = match result {
        Some(token) => InputHeadOutcome::Unsat(report, token),
        None => InputHeadOutcome::Unknown(report),
    };
    (outcome, continuation)
}

// An earlier deadline is never extended. Splitting the remaining duration
// also handles an instance/slice with less than the nominal 750 ms available.
fn directional_stage_deadline(now: Instant, root: Instant, cover: bool) -> Instant {
    if !cover {
        return root;
    }
    now.checked_add(root.saturating_duration_since(now).mul_f64(3.0 / 5.0))
        .unwrap_or(root)
        .min(root)
}

// Retain the previously completed floor if either original authority expires
// before batch commit. The helper also permits a deterministic post-completion
// expiry regression, without sleeping or pretending entry refusal is rollback.
fn commit_triangle_best(before: f64, proposed: f64, slice: Instant, authority: Instant) -> f64 {
    if live(slice) && live(authority) && fp_qualified() && good(proposed) {
        before.max(proposed)
    } else {
        before
    }
}

#[path = "input_head_bridge.rs"]
mod bridge;
pub use bridge::InputHeadContinuation;

#[path = "input_head_triangle.rs"]
mod triangle;

#[path = "input_head_directional.rs"]
mod directional;
#[path = "input_head_phase_cover.rs"]
mod phase_cover;

// Keep the complete-stage prototype executable in tests until its caller has
// an explicit original-instance deadline and ordinary verdict authorization.
#[cfg(test)]
#[path = "input_head_phase_tree.rs"]
mod phase_tree;

#[cfg(test)]
#[path = "input_head_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "input_head_triangle_integration_tests.rs"]
mod triangle_integration_tests;
