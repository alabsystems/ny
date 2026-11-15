// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Opaque continuation of a complete source-owned functional, not a generic
//! layer bound. Completed forms are restricted to contained input boxes. A
//! separate private box ledger must cover the entire evaluation source before
//! this continuation can authorize the original singleton instance.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

const MAX_FORMS: usize = 6; // baseline plus native/legacy/triangle candidates
const MAX_LEAVES: usize = 4096;
const MAX_COORDINATES: usize = 1_048_576;
const MAX_COVER_WORK: usize = 1_048_576;

const MAX_REPRICE_MASKS: usize = 256;
const MAX_REPRICE_CELLS: usize = 2_000_000;

/// One optional work ledger for the entire wider-root build, not per form.
/// Small groups reserve a conservative full-mask cost before their unchanged
/// checker; five-head groups report the same checker's actually charged work.
struct GroupRepriceBudget {
    deadline: Instant,
    masks: usize,
    cells: usize,
}
impl GroupRepriceBudget {
    fn new(deadline: Instant) -> Self {
        Self {
            deadline,
            masks: 0,
            cells: 0,
        }
    }
    fn small_group(&mut self, source: &Source, group: &[(usize, f64)]) -> Option<f64> {
        if !live(self.deadline) || !fp_qualified() || !(2..=4).contains(&group.len()) {
            return None;
        }
        let masks = 1usize.checked_shl(u32::try_from(group.len()).ok()?)?;
        let d = source.spec.inputs;
        let per_mask = d
            .checked_mul(2)?
            .checked_add(group.len().checked_mul(d.checked_add(2)?)?)?;
        let cells = per_mask.checked_mul(masks)?;
        let next_masks = self.masks.checked_add(masks)?;
        let next_cells = self.cells.checked_add(cells)?;
        if next_masks > MAX_REPRICE_MASKS || next_cells > MAX_REPRICE_CELLS {
            return None;
        }
        // Charge attempted full work even if later arithmetic/local expiry refuses it.
        self.masks = next_masks;
        self.cells = next_cells;
        source.negative_group_lower(group, self.deadline)
    }
    fn five_group(&mut self, source: &Source, group: &[(usize, f64)]) -> Option<f64> {
        triangle::reprice_five_group(
            source,
            group,
            self.deadline,
            &mut self.masks,
            &mut self.cells,
        )
    }
}

#[derive(Clone)]
pub(super) struct ResidualForm {
    pub(super) generation: Arc<InputHeadSpec>,
    pub(super) constant: I,
    pub(super) input: Vec<I>,
    pub(super) relu: Vec<I>,
}

#[derive(Clone)]
pub(super) struct RowRecipe {
    pub(super) coefficients: Vec<(usize, I)>,
    pub(super) mass: Option<triangle::MassCut>,
    pub(super) group: Option<Vec<(usize, f64)>>,
}

#[derive(Clone)]
pub(super) enum Recipe {
    Slopes(Vec<f64>),
    Dual {
        rows: Vec<RowRecipe>,
        points: Vec<f64>,
    },
}

impl ResidualForm {
    fn valid(&self, source: &Source) -> bool {
        Arc::ptr_eq(&self.generation, &source.spec)
            && self.constant.valid()
            && self.input.len() == source.spec.inputs
            && self.relu.len() == source.spec.heads
            && self.input.iter().chain(&self.relu).all(|a| a.valid())
    }

    pub(super) fn support(
        &self,
        source: &Source,
        inputs: &[I],
        relu: &[I],
        deadline: Instant,
    ) -> Option<f64> {
        if !live(deadline)
            || !fp_qualified()
            || !self.valid(source)
            || inputs.len() != self.input.len()
            || relu.len() != self.relu.len()
        {
            return None;
        }
        let mut lower = self.constant;
        for (j, (&a, &x)) in self.input.iter().zip(inputs).enumerate() {
            if j & 63 == 0 && !live(deadline) {
                return None;
            }
            lower = lower.add(a.mul(x)?)?;
        }
        for (j, (&a, &r)) in self.relu.iter().zip(relu).enumerate() {
            if j & 63 == 0 && !live(deadline) {
                return None;
            }
            // The full residual remains stored, including exact zero lanes.
            lower = lower.add(a.mul(r)?)?;
        }
        (live(deadline) && fp_qualified() && lower.valid()).then_some(lower.lo)
    }
}

impl Recipe {
    #[cfg_attr(not(test), expect(dead_code, reason = "component regression controls"))]
    fn replay(&self, source: &Source, deadline: Instant) -> Option<ResidualForm> {
        self.replay_with_budget(source, deadline, &mut GroupRepriceBudget::new(deadline))
    }

    fn replay_with_budget(
        &self,
        source: &Source,
        deadline: Instant,
        budget: &mut GroupRepriceBudget,
    ) -> Option<ResidualForm> {
        // The supplied cap can only shorten the existing shared preparation cap.
        let deadline = deadline.min(budget.deadline);
        budget.deadline = deadline;
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        match self {
            Self::Slopes(v) => source.replay_retained(v, deadline).map(|(_, f)| f),
            Self::Dual { rows, points } => {
                if rows.len() != points.len()
                    || rows.len() > ny_core::head_functional_advice::FUNCTIONAL_MAX_ROWS
                {
                    return None;
                }
                let d = source.spec.inputs;
                let n = d.checked_add(source.spec.heads)?;
                let mut residual = vec![I::p(0.0); d];
                residual.extend(source.q.iter().copied());
                let mut constant = source.bias.sub(I::p(source.output_error))?;
                for (row, &point) in rows.iter().zip(points) {
                    if !live(deadline) || !good(point) || point < 0.0 {
                        return None;
                    }
                    if point == 0.0 {
                        continue;
                    }
                    if row.coefficients.iter().enumerate().any(|(at, &(j, a))| {
                        j >= n || !a.valid() || (at > 0 && row.coefficients[at - 1].0 >= j)
                    }) || (row.mass.is_some() && row.group.is_some())
                    {
                        return None;
                    }
                    // Rebuild every RHS on the NEW evaluation Source. No old
                    // narrow-box RHS or certified endpoint is extended.
                    let rhs = match &row.mass {
                        Some(cut) => {
                            if !triangle::mass_recipe_matches(cut, &row.coefficients, d) {
                                return None;
                            }
                            triangle::source_cut_lower(source, cut, deadline)?
                        }
                        None => match &row.group {
                            Some(group) => {
                                let expected = group
                                    .iter()
                                    .filter(|(_, t)| *t != 0.0)
                                    .map(|&(j, t)| d.checked_add(j).map(|k| (k, -t)))
                                    .collect::<Option<Vec<_>>>()?;
                                if expected.len() != row.coefficients.len()
                                    || expected
                                        .iter()
                                        .zip(&row.coefficients)
                                        .any(|(&(j, t), &(k, a))| j != k || a.lo != t || a.hi != t)
                                {
                                    return None;
                                }
                                match group.len() {
                                    2..=4 => budget.small_group(source, group)?,
                                    5 => source.five_negative_group_lower(group, budget)?,
                                    _ => return None,
                                }
                            }
                            None => source.row_lhs_lower(&row.coefficients, deadline)?,
                        },
                    };
                    let p = I::p(point);
                    constant = constant.add(p.mul(I::p(rhs))?)?;
                    for (at, &(j, a)) in row.coefficients.iter().enumerate() {
                        if at & 63 == 0 && !live(deadline) {
                            return None;
                        }
                        residual[j] = residual[j].sub(p.mul(a)?)?;
                    }
                }
                let form = ResidualForm {
                    generation: Arc::clone(&source.spec),
                    constant,
                    input: residual[..d].to_vec(),
                    relu: residual[d..].to_vec(),
                };
                (live(deadline) && fp_qualified() && form.valid(source)).then_some(form)
            }
        }
    }
}

impl Source {
    /// Certify the complete sparse row LHS over this Source. This is used to
    /// reprice original triangle-row recipes after evaluation-box widening.
    /// Non-cut rows have one ReLU lane, so the ordinary epigraph/triangle
    /// cancellation costs O(D), not an all-head matrix loop per row.
    fn row_lhs_lower(&self, row: &[(usize, I)], deadline: Instant) -> Option<f64> {
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        let d = self.spec.inputs;
        let h = self.spec.heads;
        let mut input = vec![I::p(0.0); d];
        let mut constant = I::p(0.0);
        for &(index, q) in row {
            if !live(deadline) || !q.valid() {
                return None;
            }
            if index < d {
                input[index] = input[index].add(q)?;
                continue;
            }
            let j = index.checked_sub(d)?;
            if j >= h {
                return None;
            }
            let midpoint = 0.5 * q.lo + 0.5 * q.hi;
            let y = self.y[j];
            let alpha = if y.hi <= 0.0 {
                0.0
            } else if y.lo >= 0.0 {
                1.0
            } else if midpoint < 0.0 {
                y.hi / (y.hi - y.lo)
            } else {
                f64::from(u8::from(y.hi >= -y.lo))
            };
            let v = midpoint * alpha;
            if !good(v) {
                return None;
            }
            constant = constant
                .add(I::p(scalar_intercept(q, v, y)?))?
                .add(I::p(v).mul(I::p(f64::from(self.spec.first_bias[j])))?)?
                .sub(I::p(v.abs()).mul(I::p(self.first_error[j]))?)?;
            for (k, a) in input.iter_mut().enumerate() {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                *a = a.add(I::p(v).mul(I::p(f64::from(self.spec.first[k * h + j])))?)?;
            }
        }
        for (k, a) in input.iter().enumerate() {
            if k & 63 == 0 && !live(deadline) {
                return None;
            }
            constant = constant.add(a.mul(I {
                lo: self.lo[k],
                hi: self.hi[k],
            })?)?;
        }
        (live(deadline) && fp_qualified()).then_some(constant.lo)
    }

    fn negative_group_lower(&self, group: &[(usize, f64)], deadline: Instant) -> Option<f64> {
        if !live(deadline)
            || !fp_qualified()
            || !(2..=4).contains(&group.len())
            || group.iter().enumerate().any(|(at, &(j, tau))| {
                j >= self.spec.heads || !good(tau) || tau < 0.0 || (at > 0 && group[at - 1].0 >= j)
            })
        {
            return None;
        }
        let d = self.spec.inputs;
        let h = self.spec.heads;
        let mut lower = f64::INFINITY;
        for mask in 0..(1usize << group.len()) {
            let mut constant = I::p(0.0);
            let mut input = vec![I::p(0.0); d];
            for (at, &(head, tau)) in group.iter().enumerate() {
                if !live(deadline) {
                    return None;
                }
                if mask & (1 << at) == 0 {
                    continue;
                }
                let q = I::p(-tau);
                constant = constant
                    .add(q.mul(I::p(f64::from(self.spec.first_bias[head])))?)?
                    .sub(I::p(tau).mul(I::p(self.first_error[head]))?)?;
                for (k, a) in input.iter_mut().enumerate() {
                    if k & 63 == 0 && !live(deadline) {
                        return None;
                    }
                    *a = a.add(q.mul(I::p(f64::from(self.spec.first[k * h + head])))?)?;
                }
            }
            for (k, a) in input.iter().enumerate() {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                constant = constant.add(a.mul(I {
                    lo: self.lo[k],
                    hi: self.hi[k],
                })?)?;
            }
            lower = lower.min(constant.lo);
        }
        (live(deadline) && fp_qualified() && good(lower)).then_some(lower)
    }

    // Five heads are a separately authenticated source operation. The old <=4
    // body above stays literal; a fifth lane never goes through a truncated mask set.
    fn five_negative_group_lower(
        &self,
        group: &[(usize, f64)],
        budget: &mut GroupRepriceBudget,
    ) -> Option<f64> {
        if group.len() != 5 {
            return None;
        }
        budget.five_group(self, group)
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "component regression controls"))]
    fn relu_boxes_on(&self, inputs: &[I], deadline: Instant) -> Option<Vec<I>> {
        Some(child_native::ChildBox::new(self, inputs, deadline)?.relu())
    }

    fn difference_tail_error(&self, deadline: Instant) -> Option<f64> {
        let mut absolute = I::p(0.0);
        for o in [self.target, self.adversary] {
            absolute = absolute.add(I::p(f64::from(self.spec.final_bias[o]).abs()))?;
            for j in 0..self.spec.heads {
                if j & 63 == 0 && !live(deadline) {
                    return None;
                }
                absolute = absolute.add(
                    I::p(f64::from(self.spec.final_weight[j * 2 + o]).abs())
                        .mul(I::p(self.y[j].hi.max(0.0)))?,
                )?;
            }
        }
        absolute = absolute.add(I::p(self.output_error))?;
        dot_error(absolute.hi, 2)
    }
}

#[derive(Clone)]
struct Leaf {
    lo: Vec<f64>,
    hi: Vec<f64>,
    lower: f64,
}

#[derive(Default)]
struct Ledger {
    leaves: Vec<Leaf>,
    overflow: bool,
}

/// Concrete source-owned continuation. No public lower-bound, row, source,
/// residual or leaf constructor exists. A generic Verified/clip/empty queue
/// cannot fill this object's independent source-cover ledger.
pub struct InputHeadContinuation {
    source: Arc<Source>,
    forms: Vec<ResidualForm>,
    // Sealed at build over the evaluation ROOT; every child native pass
    // charges exactly the same scalar-output execution envelope once.
    tail_error: f64,
    authority: Instant,
    ledger: Mutex<Ledger>,
    dependent: AtomicBool,
    supports: AtomicUsize,
}

impl std::fmt::Debug for InputHeadContinuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputHeadContinuation")
            .field("inputs", &self.source.spec.inputs)
            .field("heads", &self.source.spec.heads)
            .field("forms", &self.forms.len())
            .field("dependent_use", &self.dependent_use())
            .finish_non_exhaustive()
    }
}

impl InputHeadContinuation {
    pub(super) fn build(
        original: &Source,
        completed: &[ResidualForm],
        recipes: &[Recipe],
        lo: &[f64],
        hi: &[f64],
        prep_deadline: Instant,
        authority: Instant,
    ) -> Option<Arc<Self>> {
        if !live(prep_deadline)
            || !live(authority)
            || !fp_qualified()
            || lo.len() != original.lo.len()
            || hi.len() != lo.len()
            || lo.iter().zip(&original.lo).any(|(&a, &b)| a > b)
            || hi.iter().zip(&original.hi).any(|(&a, &b)| a < b)
        {
            return None;
        }
        let same_box = lo == original.lo && hi == original.hi;
        // A wider generic root gets a fresh immutable Source generation and
        // fresh E1/Eout/Q/Y. Finite recipes are advice, not widened proofs.
        let source = if same_box {
            original.clone()
        } else {
            Source::build(
                &original.spec,
                lo,
                hi,
                original.target,
                original.adversary,
                prep_deadline,
            )?
        };
        let mut forms = vec![ResidualForm {
            generation: Arc::clone(&source.spec),
            constant: source.bias.sub(I::p(source.output_error))?,
            input: vec![I::p(0.0); source.spec.inputs],
            relu: source.q.clone(),
        }];
        if same_box {
            forms.extend(completed.iter().take(MAX_FORMS - 1).cloned());
        } else {
            let mut group_budget = GroupRepriceBudget::new(prep_deadline);
            for recipe in recipes.iter().take(MAX_FORMS - 1) {
                if !live(prep_deadline) {
                    break;
                }
                if let Some(form) =
                    recipe.replay_with_budget(&source, prep_deadline, &mut group_budget)
                {
                    forms.push(form);
                }
            }
            eprintln!("[input-head-bridge-group-reprice] charged_masks={} work_cells={} completed_forms={}",
                group_budget.masks, group_budget.cells, forms.len().saturating_sub(1));
        }
        // Also cover a RN/gradual scalar ±1 output-difference tail. Exact
        // original-margin authority still lives in this separate Source.
        let tail_error = source.difference_tail_error(prep_deadline)?;
        for form in &mut forms {
            form.constant = form.constant.sub(I::p(tail_error))?;
        }
        if forms.iter().any(|f| !f.valid(&source))
            || !live(prep_deadline)
            || !live(authority)
            || !fp_qualified()
        {
            return None;
        }
        Some(Arc::new(Self {
            source: Arc::new(source),
            forms,
            tail_error,
            authority,
            ledger: Mutex::new(Ledger::default()),
            dependent: AtomicBool::new(false),
            supports: AtomicUsize::new(0),
        }))
    }

    pub fn is_live(&self, deadline: Instant) -> bool {
        live(deadline.min(self.authority)) && fp_qualified()
    }

    pub fn root_box_matches(&self, lo: &[f32], hi: &[f32]) -> bool {
        lo.len() == self.source.lo.len()
            && hi.len() == lo.len()
            && lo
                .iter()
                .zip(&self.source.lo)
                .all(|(&a, &b)| a.is_finite() && f64::from(a) == b)
            && hi
                .iter()
                .zip(&self.source.hi)
                .all(|(&a, &b)| a.is_finite() && f64::from(a) == b)
    }

    /// Authenticate the ACTUAL original dense sequence and the scalar
    /// singleton tail. Fused/folded coefficients, changed bias/order, extra
    /// operators or an arbitrary one-output network are declined.
    fn original_prefix_len(&self, network: &crate::Network) -> Option<usize> {
        use crate::Layer;
        let s = &self.source.spec;
        let matrix_equal =
            |layer: &crate::layers::LinearLayer, outputs: usize, inputs: usize, values: &[f32]| {
                layer.weight().shape() == [outputs, inputs]
                    && (0..outputs).all(|o| {
                        (0..inputs).all(|i| {
                            layer.weight()[[o, i]].to_bits() == values[i * outputs + o].to_bits()
                        })
                    })
            };
        let bias_equal = |layer: &crate::layers::LinearLayer, values: &[f32]| {
            layer.bias().is_some_and(|b| {
                b.len() == values.len()
                    && b.iter()
                        .zip(values)
                        .all(|(&a, &b)| a.to_bits() == b.to_bits())
            })
        };
        let no_bias = |layer: &crate::layers::LinearLayer, outputs: usize| {
            layer
                .bias()
                .is_none_or(|b| b.len() == outputs && b.iter().all(|&x| x == 0.0))
        };
        let add_equal = |layer: &crate::layers::AddConstantLayer, values: &[f32]| {
            let a = layer.constant();
            (a.shape() == [values.len()] || a.shape() == [1, values.len()])
                && a.iter()
                    .zip(values)
                    .all(|(&a, &b)| a.to_bits() == b.to_bits())
        };
        let layers = network.layers();
        let prefix = match layers {
            [Layer::Linear(first), Layer::ReLU(_), Layer::Linear(last), tail @ ..]
                if matrix_equal(first, s.heads, s.inputs, &s.first)
                    && bias_equal(first, &s.first_bias)
                    && matrix_equal(last, 2, s.heads, &s.final_weight)
                    && bias_equal(last, &s.final_bias) =>
            {
                layers.len() - tail.len()
            }
            [Layer::Linear(first), Layer::AddConstant(b1), Layer::ReLU(_), Layer::Linear(last), Layer::AddConstant(b2), tail @ ..]
                if matrix_equal(first, s.heads, s.inputs, &s.first)
                    && no_bias(first, s.heads)
                    && add_equal(b1, &s.first_bias)
                    && matrix_equal(last, 2, s.heads, &s.final_weight)
                    && no_bias(last, 2)
                    && add_equal(b2, &s.final_bias) =>
            {
                layers.len() - tail.len()
            }
            _ => return None,
        };
        Some(prefix)
    }

    /// Exact original two-output sequence, for an independently performed
    /// whole-root fallback. This predicate itself is not a MIP verdict or
    /// receipt and cannot bless a descendant domain or a captured status.
    pub fn matches_original_network(&self, network: &crate::Network) -> bool {
        self.original_prefix_len(network)
            .is_some_and(|n| n == network.num_layers())
    }

    pub fn matches_scalar_network(&self, network: &crate::Network) -> bool {
        use crate::Layer;
        let Some(prefix) = self.original_prefix_len(network) else {
            return false;
        };
        let tail = &network.layers()[prefix..];
        let no_bias = |layer: &crate::layers::LinearLayer, outputs: usize| {
            layer
                .bias()
                .is_none_or(|b| b.len() == outputs && b.iter().all(|&x| x == 0.0))
        };
        let diff = match tail {
            // Exact singleton reduction. The objective is already scalar;
            // no pooling/reshape operator is needed to take max of one row.
            [Layer::Linear(diff)] => diff,
            // Preserve authentication of the historical singleton layout.
            [Layer::Linear(diff), Layer::Reshape(pre), Layer::MaxPool2d(pool), Layer::Reshape(post)]
                if pre.target_shape == [1, 1, 1]
                    && post.target_shape == [1]
                    && pool.kernel_size == (1, 1)
                    && pool.stride == (1, 1)
                    && pool.padding == (0, 0) =>
            {
                diff
            }
            _ => return false,
        };
        if diff.weight().shape() != [1, 2] || !no_bias(diff, 1) {
            return false;
        }
        (0..2).all(|o| diff.weight()[[0, o]] == if o == self.source.target { 1.0 } else { -1.0 })
    }

    pub fn dependent_use(&self) -> bool {
        self.dependent.load(Ordering::Acquire)
    }

    /// A control flag, never evidence. Consumers set this for positive
    /// pruning and unsupported floor uses. The authenticated nonpositive
    /// scalar MAX exception does not clear a previous dependency. Dependent
    /// closure still requires a completed private Source partition.
    pub fn mark_dependent_use(&self) {
        self.dependent.store(true, Ordering::Release);
    }

    /// Restrict complete source functionals to an actual child box. FP is
    /// qualified on THIS consuming thread before and after every arithmetic
    /// pass, including Rayon workers. A foreign/wider/expired box is refused.
    pub fn try_support_box(&self, lo: &[f32], hi: &[f32], deadline: Instant) -> Option<f32> {
        let lower = lo.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        let upper = hi.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        let (bound, inputs) = self.support(&lower, &upper, deadline)?;
        let rounded = lower_f32(bound)?;
        if !live(deadline.min(self.authority)) || !fp_qualified() {
            return None;
        }
        if rounded > 0.0 {
            let mut ledger = self.ledger.try_lock().ok()?;
            if ledger.leaves.len() >= MAX_LEAVES
                || (ledger.leaves.len() + 1)
                    .checked_mul(inputs.len())?
                    .checked_mul(2)?
                    > MAX_COORDINATES
            {
                if !self.is_live(deadline) {
                    return None;
                }
                ledger.overflow = true;
                return self.is_live(deadline).then_some(rounded);
            }
            let duplicate = ledger.leaves.iter().any(|l| l.lo == lower && l.hi == upper);
            if !self.is_live(deadline) {
                return None;
            }
            if !duplicate {
                ledger.leaves.push(Leaf {
                    lo: lower,
                    hi: upper,
                    lower: bound,
                });
            }
        }
        self.supports.fetch_add(1, Ordering::Relaxed);
        self.is_live(deadline).then_some(rounded)
    }

    fn support(&self, lo: &[f64], hi: &[f64], deadline: Instant) -> Option<(f64, Vec<I>)> {
        let deadline = deadline.min(self.authority);
        if !live(deadline)
            || !fp_qualified()
            || lo.len() != self.source.lo.len()
            || hi.len() != lo.len()
            || lo.iter().zip(hi).enumerate().any(|(k, (&l, &h))| {
                !good(l) || !good(h) || l > h || l < self.source.lo[k] || h > self.source.hi[k]
            })
        {
            return None;
        }
        let inputs: Vec<_> = lo.iter().zip(hi).map(|(&lo, &hi)| I { lo, hi }).collect();
        let child = child_native::ChildBox::new(&self.source, &inputs, deadline)?;
        let relu = child.relu();
        let mut best = f64::NEG_INFINITY;
        for form in &self.forms {
            if let Some(lower) = form.support(&self.source, &inputs, &relu, deadline) {
                best = best.max(lower);
            }
        }
        // Optional work receives only a fraction of the ACTUAL remaining
        // child/original authority. Failure/local expiry leaves completed
        // old floors bit-exact; global expiry still refuses every return.
        let proper_child = inputs
            .iter()
            .enumerate()
            .any(|(k, x)| x.lo > self.source.lo[k] || x.hi < self.source.hi[k]);
        if proper_child {
            let now = Instant::now();
            if let Some(native_deadline) = now
                .checked_add(deadline.saturating_duration_since(now) / 4)
                .map(|end| end.min(deadline))
            {
                if let Some(lower) =
                    child.native_lower(&self.source, self.tail_error, native_deadline)
                {
                    best = best.max(lower);
                }
            }
        }
        (good(best) && live(deadline) && fp_qualified()).then_some((best, inputs))
    }

    /// Pure proof-completeness query for the engine result barrier. This does
    /// not replace Instance authorization, containment or original G3.
    pub fn has_complete_source_cover(&self, deadline: Instant) -> bool {
        let deadline = deadline.min(self.authority);
        if !live(deadline) || !fp_qualified() {
            return false;
        }
        let Ok(ledger) = self.ledger.try_lock() else {
            return false;
        };
        if ledger.overflow {
            return false;
        }
        let leaves = ledger.leaves.clone();
        drop(ledger);
        complete_cover_lower(&leaves, &self.source.lo, &self.source.hi, deadline).is_some()
    }

    /// Authorize only a COMPLETE source-owned partition of the ENTIRE
    /// evaluation root. Generic pruning, clip receipts or queue exhaustion
    /// are not imported. The caller still binds raw bytes and actually runs
    /// original singleton G3 before publishing this Instance token.
    pub fn try_finish(&self, deadline: Instant) -> Option<UnsatToken> {
        let deadline = deadline.min(self.authority);
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        let ledger = self.ledger.try_lock().ok()?;
        if ledger.overflow || ledger.leaves.is_empty() {
            return None;
        }
        let leaves = ledger.leaves.clone();
        drop(ledger);
        let lower = complete_cover_lower(&leaves, &self.source.lo, &self.source.hi, deadline)?;
        let cover = crate::certify::margin_row::root_class(
            RoundMode::Outward,
            self.source.adversary,
            lower,
        )?;
        let frame = Frame {
            source: &self.source,
            lower,
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
        let token = authorize_unsat(
            TokenScope::Instance,
            &[self.source.adversary],
            &[cover],
            &g2.report(),
        )
        .ok()?;
        if !live(deadline) || !fp_qualified() {
            return None;
        }
        eprintln!("[input-head-bridge] status=complete leaves={} supports={} lower_bits={:016x} classes=1", leaves.len(), self.supports.load(Ordering::Relaxed), lower.to_bits());
        eprintln!(
            "[cover-gate] authorized scope=instance classes=1 tree=none {}",
            g2.fragment()
        );
        Some(token)
    }
}

fn lower_f32(x: f64) -> Option<f32> {
    if !good(x) || x.abs() > f64::from(f32::MAX) {
        return None;
    }
    let mut point = x as f32;
    if f64::from(point) > x {
        point = point.next_down();
    }
    point.is_finite().then_some(point)
}

/// Private geometric union check. Every member is already a positive Source
/// certificate; geometry alone cannot mint a leaf. Exact f32 endpoints widened
/// to f64 are split at existing leaf boundaries; no tolerance or epsilon is
/// used. Work/deadline exhaustion refuses the complete cover.
fn complete_cover_lower(
    leaves: &[Leaf],
    root_lo: &[f64],
    root_hi: &[f64],
    deadline: Instant,
) -> Option<f64> {
    if !live(deadline) || !fp_qualified() || leaves.is_empty() || root_lo.len() != root_hi.len() {
        return None;
    }
    let d = root_lo.len();
    if leaves.iter().any(|l| {
        l.lo.len() != d
            || l.hi.len() != d
            || !good(l.lower)
            || l.lower <= 0.0
            || (0..d).any(|k| {
                !good(l.lo[k])
                    || !good(l.hi[k])
                    || l.lo[k] > l.hi[k]
                    || l.lo[k] < root_lo[k]
                    || l.hi[k] > root_hi[k]
            })
    }) {
        return None;
    }
    let mut stack = vec![(
        root_lo.to_vec(),
        root_hi.to_vec(),
        (0..leaves.len()).collect::<Vec<_>>(),
    )];
    let mut work = 0usize;
    while let Some((lo, hi, members)) = stack.pop() {
        work = work.checked_add(members.len().checked_mul(d)?)?;
        if work > MAX_COVER_WORK || !live(deadline) {
            return None;
        }
        if members
            .iter()
            .any(|&j| (0..d).all(|k| leaves[j].lo[k] <= lo[k] && leaves[j].hi[k] >= hi[k]))
        {
            continue;
        }
        let mut cut = None;
        'search: for (at, &j) in members.iter().enumerate() {
            if at & 63 == 0 && !live(deadline) {
                return None;
            }
            for k in 0..d {
                for point in [leaves[j].lo[k], leaves[j].hi[k]] {
                    if point > lo[k] && point < hi[k] {
                        cut = Some((k, point));
                        break 'search;
                    }
                }
            }
        }
        let (k, point) = cut?;
        let mut left_hi = hi.clone();
        left_hi[k] = point;
        let mut right_lo = lo.clone();
        right_lo[k] = point;
        let relevant = |j: usize, a: &[f64], b: &[f64]| {
            (0..d).all(|k| {
                if a[k] == b[k] {
                    leaves[j].lo[k] <= a[k] && leaves[j].hi[k] >= b[k]
                } else {
                    leaves[j].lo[k] < b[k] && leaves[j].hi[k] > a[k]
                }
            })
        };
        let left = members
            .iter()
            .copied()
            .filter(|&j| relevant(j, &lo, &left_hi))
            .collect();
        let right = members
            .iter()
            .copied()
            .filter(|&j| relevant(j, &right_lo, &hi))
            .collect();
        stack.push((right_lo, hi, right));
        stack.push((lo, left_hi, left));
    }
    (live(deadline) && fp_qualified())
        .then(|| leaves.iter().map(|l| l.lower).fold(f64::INFINITY, f64::min))
}

#[cfg(test)]
#[path = "input_head_bridge_tests.rs"]
mod tests;

// Child intercepts are source/domain owned and never cached as ROOT forms.
#[path = "input_head_child_native.rs"]
mod child_native;

#[cfg(test)]
#[path = "input_head_bridge_five_negative_groups_tests.rs"]
mod five_negative_group_tests;
