// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Predicate execution over binary32 effective inputs and outputs.
//!
//! The pinned organizer evaluates expressions containing numpy.float32 values
//! in binary32, including weak-scalar promotion of ordinary numeric literals.
//! Real arithmetic is not interchangeable with this: a nonzero product may
//! round to zero. This separate profile keeps the existing real evaluators.
//! A caller must authenticate that every supplied variable is binary32 (for
//! example from a checked original Float32 ONNX execution enclosure). Merely
//! declaring a real variable does NOT establish that execution contract.
use super::*;
use std::time::Instant;

const MAX_NODES: usize = 2048;
const MAX_VARIABLES: usize = 16;

/// A bounded plan borrowed from the complete original predicate. It cannot
/// authorize a network enclosure or publish a solver verdict by itself.
pub struct PreparedF32Predicate<'a> {
    formula: &'a NonLinearFormula,
}

struct Poll<'a> {
    end: Instant,
    callback: &'a mut dyn FnMut() -> bool,
    alive: bool,
}
impl Poll<'_> {
    fn check(&mut self) -> Option<()> {
        self.alive &= Instant::now() < self.end && (self.callback)();
        self.alive.then_some(())
    }
}

// Keep nominal binary32 operations separate, including cancellation/underflow.
#[inline(never)]
fn op32(op: &str, a: f32, b: f32) -> f32 {
    let (a, b) = (std::hint::black_box(a), std::hint::black_box(b));
    match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        _ => f32::NAN,
    }
}
pub(super) fn environment() -> bool {
    has_f64_interval_proof_environment()
        && op32("+", 1.0, f32::from_bits(0x3380_0000)).to_bits() == 1.0f32.to_bits()
        && op32("*", f32::from_bits(1), 1.0).to_bits() == 1
        && op32("*", f32::from_bits(1), 0.5).to_bits() == 0
        && op32("*", f32::from_bits(3), 0.5).to_bits() == 2
}

/// Nearest rounding is monotone. These endpoints enclose every binary32
/// conversion of a value in the input interval. Openness is deliberately lost:
/// a strict positive real value can round to zero.
fn rounded(iv: OpenIv) -> Option<OpenIv> {
    let (lo, hi) = (iv.lo as f32, iv.hi as f32);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    OpenIv::closed(f64::from(lo), f64::from(hi)).checked()
}

/// Intersection of a real product piece with the finite binary32 grid. This
/// is narrowing a KNOWN binary32 variable, not rounding a real input image.
/// `Some(None)` is an empty piece, never a malformed-domain refusal.
#[expect(
    clippy::option_option,
    reason = "None refuses malformed intervals; Some(None) proves the binary32 intersection is empty."
)]
fn grid(iv: OpenIv) -> Option<Option<OpenIv>> {
    iv.checked()?;
    let (mut lo, mut hi) = (iv.lo as f32, iv.hi as f32);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    if f64::from(lo) < iv.lo || (iv.lo_open && f64::from(lo) == iv.lo) {
        lo = lo.next_up();
    }
    if f64::from(hi) > iv.hi || (iv.hi_open && f64::from(hi) == iv.hi) {
        hi = hi.next_down();
    }
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Some(None);
    }
    Some(Some(OpenIv::closed(f64::from(lo), f64::from(hi))))
}

#[derive(Clone, Copy)]
struct NumberIv {
    iv: OpenIv,
    binary32: bool,
}
impl NumberIv {
    fn combine(self, op: &str, rhs: Self) -> Option<Self> {
        let binary32 = self.binary32 || rhs.binary32;
        let (a, b) = if binary32 {
            (rounded(self.iv)?, rounded(rhs.iv)?)
        } else {
            (self.iv, rhs.iv)
        };
        let pairs = match op {
            "+" => [(a.lo, b.lo), (a.hi, b.hi), (a.lo, b.lo), (a.hi, b.hi)],
            "-" => [(a.lo, b.hi), (a.hi, b.lo), (a.lo, b.hi), (a.hi, b.lo)],
            "*" => [(a.lo, b.lo), (a.lo, b.hi), (a.hi, b.lo), (a.hi, b.hi)],
            _ => return None,
        };
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (x, y) in pairs {
            let value = if binary32 {
                f64::from(op32(op, x as f32, y as f32))
            } else {
                match op {
                    "+" => x + y,
                    "-" => x - y,
                    "*" => x * y,
                    _ => return None,
                }
            };
            if !value.is_finite() {
                return None;
            }
            lo = lo.min(value);
            hi = hi.max(value);
        }
        // Binary64 constant expressions may be Python integer arithmetic as
        // well as float arithmetic. The outward hull covers both, without
        // relying on open-sign identities across floating-point underflow.
        let iv = if binary32 {
            OpenIv::closed(lo, hi)
        } else {
            let (lo, hi) = out((lo, hi))?;
            OpenIv::closed(lo, hi)
        }
        .checked()?;
        Some(Self { iv, binary32 })
    }
}

impl NonLinearFormula {
    /// Prepare a separate binary32 predicate profile. Source declarations must
    /// be one input/output tensor pair of one network, using float32 or real.
    /// Actual binary32 variable provenance remains the caller's obligation.
    pub fn prepare_f32_predicate(
        &self,
        end: Instant,
        callback: &mut dyn FnMut() -> bool,
    ) -> Option<PreparedF32Predicate<'_>> {
        let mut poll = Poll {
            end,
            callback,
            alive: true,
        };
        poll.check()?;
        if !environment()
            || !self.single_network
            || !self.f32_declaration_profile
            || self.decls.len() != 2
            || self.num_inputs + self.num_outputs > MAX_VARIABLES
            || self
                .decls
                .iter()
                .filter(|d| d.kind == VarKind::Input)
                .count()
                != 1
            || self
                .decls
                .iter()
                .filter(|d| d.kind == VarKind::Output)
                .count()
                != 1
            || self
                .decls
                .iter()
                .any(|d| !matches!(d.dtype.as_str(), "float32" | "real"))
        {
            return None;
        }
        fn walk(
            f: &NonLinearFormula,
            e: &Expr,
            boolean: bool,
            count: &mut usize,
            depth: usize,
            poll: &mut Poll<'_>,
        ) -> Option<()> {
            poll.check()?;
            *count = count.checked_add(1)?;
            if *count > MAX_NODES || depth > 64 {
                return None;
            }
            match e {
                Expr::Number(n) if !boolean => {
                    if !n.lower.is_finite()
                        || !n.upper.is_finite()
                        || n.exact.numer().bits() > 512
                        || n.exact.denom().bits() > 512
                    {
                        return None;
                    }
                }
                Expr::Symbol(s) if !boolean => {
                    if s.len() > 1024 || f.var_of(e).is_none() {
                        return None;
                    }
                }
                Expr::List(items) => {
                    if items.is_empty() || items.len() > 65 {
                        return None;
                    }
                    let Expr::Symbol(op) = &items[0] else {
                        return None;
                    };
                    let (child_boolean, arity) = if boolean {
                        match op.as_str() {
                            "and" | "or" => (true, None),
                            "not" => (true, Some(1)),
                            "<=" | "<" | ">=" | ">" | "=" | "==" | "!=" => (false, Some(2)),
                            _ => return None,
                        }
                    } else {
                        if !matches!(op.as_str(), "+" | "-" | "*") || items.len() < 2 {
                            return None;
                        }
                        (false, None)
                    };
                    if arity.is_some_and(|n| items.len() != n + 1) {
                        return None;
                    }
                    for item in &items[1..] {
                        walk(f, item, child_boolean, count, depth + 1, poll)?;
                    }
                }
                _ => return None,
            }
            Some(())
        }
        let mut count = 0;
        for a in &self.asserts {
            walk(self, a, true, &mut count, 0, &mut poll)?;
        }
        // Existing bounded pivot collection is over this same borrowed source.
        let _ = self.case_split_pivots();
        poll.check()?;
        Some(PreparedF32Predicate { formula: self })
    }
}

impl PreparedF32Predicate<'_> {
    /// Refute only after the whole binary32 product cover is checked. The
    /// numerical box must enclose the actual effective Float32 variables.
    /// A late/cancelled/unsupported operation returns None; no partial cover
    /// and no definitely-true piece can authorize publication.
    pub fn refute_box(
        &self,
        xl: &[f64],
        xu: &[f64],
        yl: &[f64],
        yu: &[f64],
        end: Instant,
        callback: &mut dyn FnMut() -> bool,
    ) -> Option<CaseSplitVerdict> {
        let mut poll = Poll {
            end,
            callback,
            alive: true,
        };
        poll.check()?;
        let f = self.formula;
        if !environment()
            || xl.len() != f.num_inputs
            || xu.len() != f.num_inputs
            || yl.len() != f.num_outputs
            || yu.len() != f.num_outputs
        {
            return None;
        }
        let make = |l: &[f64], u: &[f64]| -> Option<Vec<OpenIv>> {
            l.iter()
                .zip(u)
                .map(|(&a, &b)| OpenIv::closed(a, b).checked())
                .collect()
        };
        let base_x = make(xl, xu)?;
        let base_y = make(yl, yu)?;
        let mut axes = Vec::new();
        for set in f.case_split_pivots() {
            poll.check()?;
            let cell = match set.kind {
                VarKind::Input => *base_x.get(set.index)?,
                VarKind::Output => *base_y.get(set.index)?,
            };
            let parts = partition_at_pivots(cell, &set.values);
            if parts.is_empty() {
                return None;
            }
            axes.push((set.kind, set.index, parts));
        }
        let Some(total) = axes
            .iter()
            .try_fold(1usize, |n, (_, _, p)| n.checked_mul(p.len()))
            .filter(|&n| n <= MAX_CASE_SPLIT_PIECES)
        else {
            poll.check()?;
            return Some(CaseSplitVerdict {
                truth: Tri::Unknown,
                pieces: 0,
            });
        };
        let mut digits = vec![0usize; axes.len()];
        let mut done = 0;
        loop {
            poll.check()?;
            let mut xs = base_x.clone();
            let mut ys = base_y.clone();
            for ((kind, index, parts), &digit) in axes.iter().zip(&digits) {
                match kind {
                    VarKind::Input => xs[*index] = parts[digit],
                    VarKind::Output => ys[*index] = parts[digit],
                }
            }
            let mut empty = false;
            for cell in xs.iter_mut().chain(ys.iter_mut()) {
                match grid(*cell)? {
                    Some(value) => *cell = value,
                    None => empty = true,
                }
            }
            done += 1;
            if !empty {
                let mut false_piece = false;
                for a in &f.asserts {
                    if self.boolean(a, &xs, &ys, &mut poll)? == Tri::False {
                        false_piece = true;
                        break;
                    }
                }
                if !false_piece {
                    poll.check()?;
                    return Some(CaseSplitVerdict {
                        truth: Tri::Unknown,
                        pieces: done,
                    });
                }
            }
            let mut i = digits.len();
            loop {
                if i == 0 {
                    poll.check()?;
                    return (done == total).then_some(CaseSplitVerdict {
                        truth: Tri::False,
                        pieces: done,
                    });
                }
                i -= 1;
                digits[i] += 1;
                if digits[i] < axes[i].2.len() {
                    break;
                }
                digits[i] = 0;
            }
        }
    }

    fn numeric(
        &self,
        e: &Expr,
        x: &[OpenIv],
        y: &[OpenIv],
        poll: &mut Poll<'_>,
    ) -> Option<NumberIv> {
        poll.check()?;
        match e {
            Expr::Number(n) => Some(NumberIv {
                iv: OpenIv::closed(n.lower, n.upper).checked()?,
                binary32: false,
            }),
            Expr::Symbol(_) => {
                let (kind, i) = self.formula.var_of(e)?;
                Some(NumberIv {
                    iv: match kind {
                        VarKind::Input => *x.get(i)?,
                        VarKind::Output => *y.get(i)?,
                    },
                    binary32: true,
                })
            }
            Expr::List(items) => {
                let Expr::Symbol(op) = items.first()? else {
                    return None;
                };
                let args = &items[1..];
                let first = self.numeric(args.first()?, x, y, poll)?;
                if op == "-" && args.len() == 1 {
                    return Some(NumberIv {
                        iv: first.iv.neg(),
                        binary32: first.binary32,
                    });
                }
                // The reference Minus is head - sum(rest), not repeated rounded
                // subtraction. Preserve that evaluation order exactly.
                if op == "-" {
                    let mut tail = self.numeric(args.get(1)?, x, y, poll)?;
                    for a in &args[2..] {
                        tail = tail.combine("+", self.numeric(a, x, y, poll)?)?;
                    }
                    return first.combine("-", tail);
                }
                if !matches!(op.as_str(), "+" | "*") {
                    return None;
                }
                args[1..].iter().try_fold(first, |value, a| {
                    value.combine(op, self.numeric(a, x, y, poll)?)
                })
            }
        }
    }
    fn boolean(&self, e: &Expr, x: &[OpenIv], y: &[OpenIv], poll: &mut Poll<'_>) -> Option<Tri> {
        poll.check()?;
        let Expr::List(items) = e else {
            return None;
        };
        let Expr::Symbol(op) = items.first()? else {
            return None;
        };
        let args = &items[1..];
        match op.as_str() {
            "and" | "or" => {
                let is_and = op == "and";
                let mut unknown = false;
                for a in args {
                    match self.boolean(a, x, y, poll)? {
                        Tri::False if is_and => return Some(Tri::False),
                        Tri::True if !is_and => return Some(Tri::True),
                        Tri::Unknown => unknown = true,
                        _ => {}
                    }
                }
                Some(if unknown {
                    Tri::Unknown
                } else if is_and {
                    Tri::True
                } else {
                    Tri::False
                })
            }
            "not" => {
                let [a] = args else {
                    return None;
                };
                Some(match self.boolean(a, x, y, poll)? {
                    Tri::True => Tri::False,
                    Tri::False => Tri::True,
                    Tri::Unknown => Tri::Unknown,
                })
            }
            "<=" | "<" | ">=" | ">" | "=" | "==" | "!=" => {
                let [a, b] = args else {
                    return None;
                };
                let a = self.numeric(a, x, y, poll)?;
                let b = self.numeric(b, x, y, poll)?;
                let (a, b) = if a.binary32 || b.binary32 {
                    (rounded(a.iv)?, rounded(b.iv)?)
                } else {
                    (a.iv, b.iv)
                };
                compare_open(op, a, b)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "f32_predicate_tests.rs"]
mod tests;
