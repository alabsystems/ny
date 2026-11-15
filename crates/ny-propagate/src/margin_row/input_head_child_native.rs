// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private child-domain native functionals. No Source is cloned, rewritten,
//! or given smaller execution error. A form belongs to exactly this borrowed
//! original Source AND the complete actual child domain that priced its k.

use super::*;

const MAX_CELLS: usize = 2_000_000;

pub(super) struct ChildBox<'a> {
    root: &'a Source,
    inputs: Vec<I>,
    y: Vec<I>,
}

impl<'a> ChildBox<'a> {
    pub(super) fn new(root: &'a Source, inputs: &[I], deadline: Instant) -> Option<Self> {
        let d = root.spec.inputs;
        let h = root.spec.heads;
        let cells = d
            .checked_mul(h)?
            .checked_mul(2)?
            .checked_add(h.checked_mul(8)?)?
            .checked_add(d.checked_mul(3)?)?;
        if !live(deadline)
            || !fp_qualified()
            || cells > MAX_CELLS
            || inputs.len() != d
            || root.lo.len() != d
            || root.hi.len() != d
            || root.spec.first.len() != d.checked_mul(h)?
            || root.spec.first_bias.len() != h
            || root.first_error.len() != h
            || root.q.len() != h
            || root.y.len() != h
            || !root.bias.valid()
            || !good(root.output_error)
            || root.output_error < 0.0
            || root.target >= 2
            || root.adversary >= 2
            || root.target == root.adversary
            || root.first_error.iter().any(|&e| !good(e) || e < 0.0)
            || root.q.iter().chain(&root.y).any(|q| !q.valid())
            || inputs.iter().enumerate().any(|(k, x)| {
                !x.valid()
                    || !good(root.lo[k])
                    || !good(root.hi[k])
                    || root.lo[k] > root.hi[k]
                    || x.lo < root.lo[k]
                    || x.hi > root.hi[k]
            })
        {
            return None;
        }
        let mut y = Vec::with_capacity(h);
        for j in 0..h {
            if !live(deadline) {
                return None;
            }
            let mut interval = I::p(f64::from(root.spec.first_bias[j]));
            for (k, &x) in inputs.iter().enumerate() {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                interval = interval.add(I::p(f64::from(root.spec.first[k * h + j])).mul(x)?)?;
            }
            // ROOT input-cast/dot/bias error is valid on every contained
            // child. It is retained literally, not reestimated downward.
            y.push(interval.add(I {
                lo: -root.first_error[j],
                hi: root.first_error[j],
            })?);
        }
        (live(deadline) && fp_qualified()).then_some(Self {
            root,
            inputs: inputs.to_vec(),
            y,
        })
    }

    pub(super) fn relu(&self) -> Vec<I> {
        self.y
            .iter()
            .map(|y| I {
                lo: y.lo.max(0.0),
                hi: y.hi.max(0.0),
            })
            .collect()
    }

    fn slopes(&self, deadline: Instant) -> Option<Vec<f64>> {
        let mut v = Vec::with_capacity(self.y.len());
        for (&q, &y) in self.root.q.iter().zip(&self.y) {
            if !live(deadline) {
                return None;
            }
            let point = 0.5 * q.lo + 0.5 * q.hi;
            let alpha = if y.hi <= 0.0 {
                0.0
            } else if y.lo >= 0.0 {
                1.0
            } else if point < 0.0 {
                y.hi / (y.hi - y.lo)
            } else {
                f64::from(u8::from(y.hi >= -y.lo))
            };
            let slope = point * alpha;
            if !good(slope) {
                return None;
            }
            // Rounded ratios/products are finite proposals only. The full
            // original Q and exact represented slope are repriced below.
            v.push(slope);
        }
        (v.len() == self.root.spec.heads && live(deadline) && fp_qualified()).then_some(v)
    }

    fn replay<'b>(
        &'b self,
        root: &Source,
        slopes: &[f64],
        tail_error: f64,
        deadline: Instant,
    ) -> Option<ChildForm<'a, 'b>> {
        if !std::ptr::eq(self.root, root)
            || !Arc::ptr_eq(&self.root.spec, &root.spec)
            || slopes.len() != root.spec.heads
            || self.y.len() != root.spec.heads
            || !live(deadline)
            || !fp_qualified()
            || !good(tail_error)
            || tail_error < 0.0
        {
            return None;
        }
        let mut constant = root
            .bias
            .sub(I::p(root.output_error))?
            .sub(I::p(tail_error))?;
        let mut input = vec![I::p(0.0); root.spec.inputs];
        for (j, &v) in slopes.iter().enumerate() {
            if !live(deadline) || !good(v) {
                return None;
            }
            constant = constant
                .add(I::p(scalar_intercept(root.q[j], v, self.y[j])?))?
                .add(I::p(v).mul(I::p(f64::from(root.spec.first_bias[j])))?)?
                .sub(I::p(v.abs()).mul(I::p(root.first_error[j]))?)?;
            for (k, a) in input.iter_mut().enumerate() {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                *a =
                    a.add(I::p(v).mul(I::p(f64::from(root.spec.first[k * root.spec.heads + j])))?)?;
            }
            #[cfg(test)]
            if EXPIRE_AFTER_HEAD.with(|fault| {
                if fault.get() == Some(j + 1) {
                    fault.set(None);
                    true
                } else {
                    false
                }
            }) {
                // Deterministic completed-prefix fault: optional replay
                // observes its own expired cap, while original authority
                // and the already completed old portfolio remain live.
                let expired = Instant::now().checked_sub(Duration::from_secs(1))?;
                if !live(expired) {
                    return None;
                }
            }
        }
        (live(deadline) && fp_qualified()).then_some(ChildForm {
            child: self,
            constant,
            input,
        })
    }

    pub(super) fn native_lower(
        &self,
        root: &Source,
        tail_error: f64,
        deadline: Instant,
    ) -> Option<f64> {
        #[cfg(test)]
        if REFUSE_NATIVE.with(|fault| fault.replace(false)) {
            return None;
        }
        let slopes = self.slopes(deadline)?;
        self.replay(root, &slopes, tail_error, deadline)?
            .support(root, &self.inputs, deadline)
    }
}

/// Child intercepts never enter the root's retained-form list. Even an equal
/// network with a foreign Source borrow or a wider/different box is refused.
struct ChildForm<'a, 'b> {
    child: &'b ChildBox<'a>,
    constant: I,
    input: Vec<I>,
}
impl ChildForm<'_, '_> {
    fn support(&self, root: &Source, inputs: &[I], deadline: Instant) -> Option<f64> {
        if !live(deadline)
            || !fp_qualified()
            || !std::ptr::eq(root, self.child.root)
            || !Arc::ptr_eq(&root.spec, &self.child.root.spec)
            || !self.constant.valid()
            || self.input.len() != root.spec.inputs
            || inputs.len() != self.child.inputs.len()
            || inputs
                .iter()
                .zip(&self.child.inputs)
                .any(|(a, b)| a.lo.to_bits() != b.lo.to_bits() || a.hi.to_bits() != b.hi.to_bits())
        {
            return None;
        }
        let mut value = self.constant;
        for (k, (&a, &x)) in self.input.iter().zip(inputs).enumerate() {
            if k & 63 == 0 && !live(deadline) {
                return None;
            }
            value = value.add(a.mul(x)?)?;
        }
        (live(deadline) && fp_qualified() && value.valid()).then_some(value.lo)
    }
}

#[cfg(test)]
thread_local! {
    static REFUSE_NATIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static EXPIRE_AFTER_HEAD: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
#[path = "input_head_child_native_tests.rs"]
mod tests;
