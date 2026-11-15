// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One owner covers the full input domain under both predicate profiles.
use super::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Observations never grant proof authority, including a nonzero refuted count.
#[derive(Default, Debug)]
pub struct Float32ChainAttempt {
    pub visited: usize,
    pub split: usize,
    pub dual_refuted: usize,
    pub real_open_refuted: usize,
    pub f32_pieces: usize,
    pub shared_scalar_heads: usize,
    pub completed: bool,
}

/// Produced only after every leaf of the original domain has been refuted
/// under both the exact-real formula and the typed Float32 predicate.
pub struct Float32ChainRefutation {
    _complete_cover: (),
}

struct Cell {
    priority: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
    depth: u32,
}
impl PartialEq for Cell {
    fn eq(&self, rhs: &Self) -> bool {
        self.priority.to_bits() == rhs.priority.to_bits()
    }
}
impl Eq for Cell {}
impl PartialOrd for Cell {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Cell {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.priority.total_cmp(&rhs.priority)
    }
}
fn priority(lo: &[f64], hi: &[f64]) -> f64 {
    lo.iter().zip(hi).map(|(l, h)| h - l).sum()
}

impl PreparedFloat32Chain {
    /// Refute one complete original formula, never a relaxed output property.
    /// No result is published for an incomplete cover, late return, unsupported
    /// arithmetic, cancelled computation, or resource refusal. All accounting
    /// belongs to this invocation and shares the caller's absolute deadline.
    pub fn refute_formula(
        &self,
        formula: &NonLinearFormula,
        end: Instant,
        callback: &mut dyn FnMut() -> bool,
        attempt: &mut Float32ChainAttempt,
    ) -> Option<Float32ChainRefutation> {
        *attempt = Float32ChainAttempt::default();
        let mut poll = Poll { end, callback };
        poll.check()?;
        let predicate = formula.prepare_f32_predicate(end, poll.callback)?;
        if formula.num_inputs == 0
            || formula.num_inputs > 4
            || formula.decls.iter().any(|d| {
                d.shape
                    != match d.kind {
                        VarKind::Input => self.chain.input_shape.clone(),
                        VarKind::Output => self.chain.output_shape.clone(),
                    }
            })
        {
            return None;
        }
        let (mut lo, mut hi) = formula.input_box()?;
        // input_box uses only direct variable/constant comparisons under AND.
        // The hull with rounded endpoints therefore also covers every Float32
        // variable satisfying those same comparisons after weak promotion.
        for (l, h) in lo.iter_mut().zip(&mut hi) {
            poll.check()?;
            *l = l.min(f64::from(*l as f32));
            *h = h.max(f64::from(*h as f32));
            if !l.is_finite() || !h.is_finite() {
                return None;
            }
        }
        let mut queue = BinaryHeap::new();
        queue.push(Cell {
            priority: priority(&lo, &hi),
            lo,
            hi,
            depth: 0,
        });
        const MAX_VISITED: usize = 1 << 18;
        const MAX_PENDING: usize = 1 << 15;
        while let Some(cell) = queue.pop() {
            poll.check()?;
            if attempt.visited >= MAX_VISITED || queue.len() >= MAX_PENDING {
                return None;
            }
            attempt.visited += 1;
            let output = self.enclose(&cell.lo, &cell.hi, end, poll.callback)?;
            attempt.shared_scalar_heads += output.shared_scalar_heads;
            let mut real_false =
                formula.holds_over_box(&cell.lo, &cell.hi, &output.lower, &output.upper)?
                    == Tri::False;
            poll.check()?;
            if !real_false {
                let r = formula.holds_over_box_split(
                    &cell.lo,
                    &cell.hi,
                    &output.lower,
                    &output.upper,
                )?;
                poll.check()?;
                if r.truth == Tri::False {
                    real_false = true;
                    attempt.real_open_refuted += 1;
                }
            }
            if real_false {
                let r = predicate.refute_box(
                    &cell.lo,
                    &cell.hi,
                    &output.lower,
                    &output.upper,
                    end,
                    poll.callback,
                )?;
                attempt.f32_pieces += r.pieces;
                poll.check()?;
                if r.truth == Tri::False {
                    attempt.dual_refuted += 1;
                    continue;
                }
            }
            if cell.depth >= 200 {
                return None;
            }
            let axis = (0..cell.lo.len())
                .max_by(|&a, &b| (cell.hi[a] - cell.lo[a]).total_cmp(&(cell.hi[b] - cell.lo[b])))?;
            let mid = cell.lo[axis].midpoint(cell.hi[axis]);
            if !mid.is_finite() || mid <= cell.lo[axis] || mid >= cell.hi[axis] {
                return None;
            }
            let mut left_hi = cell.hi.clone();
            left_hi[axis] = mid;
            let mut right_lo = cell.lo.clone();
            right_lo[axis] = mid;
            queue.push(Cell {
                priority: priority(&cell.lo, &left_hi),
                lo: cell.lo,
                hi: left_hi,
                depth: cell.depth + 1,
            });
            queue.push(Cell {
                priority: priority(&right_lo, &cell.hi),
                lo: right_lo,
                hi: cell.hi,
                depth: cell.depth + 1,
            });
            attempt.split += 1;
        }
        poll.check()?;
        if attempt.visited == 0 || attempt.dual_refuted != attempt.split + 1 {
            return None;
        }
        attempt.completed = true;
        Some(Float32ChainRefutation {
            _complete_cover: (),
        })
    }
}
