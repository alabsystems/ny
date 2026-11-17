// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cache directed source errors and per-class coefficient endpoints for the
//! legacy composer. Borrowed sources stay immutable; payload is capped at 8 MiB.
//! Preparation refusal uses the original allocation-free calculation. The
//! cooperative path retains its calculation and resource checkpoints.
//!
//! Each eta starts with the original eb and folds input coordinates in the
//! original ascending order. Each class's four coefficient endpoints use the
//! original directed helpers. All remaining arithmetic retains its original
//! order. This is an evaluation schedule change, not a new rounding bound.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const MAX_BYTES: usize = 8 * 1024 * 1024;

pub(super) struct Prepared<'a> {
    root: &'a RootGates,
    mb: &'a MarginBatch,
    gates: &'a HeadGates,
    al: &'a PassOut,
    au: &'a PassOut,
    a: &'a [f64],
    u: &'a [f64],
    eta: Vec<[f64; 2]>,
    coefficients: Vec<[f64; 4]>,
}

impl<'a> Prepared<'a> {
    pub(super) fn new(
        root: &'a RootGates,
        mb: &'a MarginBatch,
        gates: &'a HeadGates,
        al: &'a PassOut,
        au: &'a PassOut,
    ) -> Result<Self> {
        #[cfg(test)]
        observe(0);
        let result = Self::with_limit(root, mb, gates, al, au, MAX_BYTES);
        #[cfg(test)]
        if result.is_ok() {
            observe(1);
        }
        result
    }

    fn with_limit(
        root: &'a RootGates,
        mb: &'a MarginBatch,
        gates: &'a HeadGates,
        al: &'a PassOut,
        au: &'a PassOut,
        limit: usize,
    ) -> Result<Self> {
        head_qualify(RoundMode::Outward)?;
        head_shapes(mb, gates)?;
        let (n, ny) = (root.mid.len(), mb.n_y);
        head_pack_shape(root, al, ny)?;
        head_pack_shape(root, au, ny)?;
        let bytes = ny
            .checked_mul(6 * size_of::<f64>())
            .ok_or_else(|| head_invalid("head confirmation cache size overflow"))?;
        if bytes > limit {
            return Err(head_invalid("head confirmation cache size limit"));
        }
        let ae =
            al.e.as_ref()
                .and_then(|e| e.as_slice())
                .ok_or_else(|| head_invalid("head confirmation lower error plane"))?;
        let ue =
            au.e.as_ref()
                .and_then(|e| e.as_slice())
                .ok_or_else(|| head_invalid("head confirmation upper error plane"))?;
        let a =
            al.a.as_slice()
                .ok_or_else(|| head_invalid("head confirmation lower layout"))?;
        let u =
            au.a.as_slice()
                .ok_or_else(|| head_invalid("head confirmation upper layout"))?;
        let check = || Ok(());
        let mut poll = HeadPoll::new(&check)?;
        let mut eta = head_buffer(ny, [0.0; 2], &mut poll)?;
        let coefficients = head_buffer(ny, [0.0; 4], &mut poll)?;
        for k in 0..n {
            if !root.lo[k].is_finite() || !root.hi[k].is_finite() || root.lo[k] > root.hi[k] {
                return Err(head_invalid("head confirmation original input endpoints"));
            }
        }
        for j in 0..ny {
            if !al.eb[j].is_finite() || al.eb[j] < 0.0 || !au.eb[j].is_finite() || au.eb[j] < 0.0 {
                return Err(head_invalid("head confirmation invalid bias errors"));
            }
            let (mut l, mut h) = (al.eb[j], au.eb[j]);
            for k in 0..n {
                let (el, eu) = (ae[k * ny + j], ue[k * ny + j]);
                if !el.is_finite() || el < 0.0 || !eu.is_finite() || eu < 0.0 {
                    return Err(head_invalid("head confirmation coefficient error"));
                }
                let xa = root.lo[k].abs().max(root.hi[k].abs());
                l = head_sum_hi(l, head_prod_hi(el, xa));
                h = head_sum_hi(h, head_prod_hi(eu, xa));
            }
            // Do not suppress nonfinite errors even for zero coefficients.
            head_finite(l)?;
            head_finite(h)?;
            eta[j] = [l, h];
        }
        Ok(Self {
            root,
            mb,
            gates,
            al,
            au,
            a,
            u,
            eta,
            coefficients,
        })
    }

    pub(super) fn row(&mut self, column: usize) -> Result<f64> {
        // Recheck after thread migration or a change to FP control registers.
        head_qualify(RoundMode::Outward)?;
        #[cfg(test)]
        observe(2);
        let Self {
            root,
            mb,
            gates,
            al,
            au,
            a,
            u,
            eta,
            coefficients,
        } = self;
        let (n, ny) = (root.mid.len(), mb.n_y);
        if column >= mb.nf() {
            return Err(head_invalid("head confirmation column shape"));
        }
        let wp = &mb.wp[column * ny..(column + 1) * ny];
        let wn = &mb.wn[column * ny..(column + 1) * ny];
        for j in 0..ny {
            if !wp[j].is_finite()
                || wp[j] < 0.0
                || !wn[j].is_finite()
                || wn[j] > 0.0
                || !gates.alpha[j].is_finite()
                || !(0.0..=1.0).contains(&gates.alpha[j])
                || !gates.s[j].is_finite()
                || !(0.0..=1.0).contains(&gates.s[j])
                || !gates.c[j].is_finite()
                || gates.c[j] < 0.0
            {
                return Err(head_invalid("head confirmation invalid source lanes"));
            }
            // Underflowed products may straddle zero. Never clamp endpoints.
            coefficients[j] = [
                head_prod_lo(wp[j], gates.alpha[j]),
                head_prod_hi(wp[j], gates.alpha[j]),
                head_prod_lo(wn[j], gates.s[j]),
                head_prod_hi(wn[j], gates.s[j]),
            ];
        }
        let mut input = 0.0;
        for k in 0..n {
            let (mut lo, mut hi) = (0.0, 0.0);
            for j in 0..ny {
                let [cal, cau, cbl, cbu] = coefficients[j];
                let (ll, lh) = head_interval_product(cal, cau, a[k * ny + j], a[k * ny + j]);
                let (ul, uh) = head_interval_product(cbl, cbu, u[k * ny + j], u[k * ny + j]);
                lo = head_sum_lo(lo, head_sum_lo(ll, ul));
                hi = head_sum_hi(hi, head_sum_hi(lh, uh));
            }
            let (lower, _) = head_interval_product(lo, hi, root.lo[k], root.hi[k]);
            input = head_sum_lo(input, lower);
        }
        let (mut bias, mut constant, mut penalty) = (0.0, 0.0, 0.0);
        for j in 0..ny {
            let [cal, cau, cbl, cbu] = coefficients[j];
            let (lower_bias, _) = head_interval_product(cal, cau, al.b[j], al.b[j]);
            let (upper_bias, _) = head_interval_product(cbl, cbu, au.b[j], au.b[j]);
            bias = head_sum_lo(bias, head_sum_lo(lower_bias, upper_bias));
            constant = head_sum_lo(constant, head_prod_lo(wn[j], gates.c[j]));
            let el = head_prod_hi(cal.abs().max(cau.abs()), eta[j][0]);
            let eu = head_prod_hi(cbl.abs().max(cbu.abs()), eta[j][1]);
            penalty = head_sum_hi(penalty, head_sum_hi(el, eu));
        }
        let lower = head_sum_lo(
            head_sum_lo(
                head_sum_lo(head_sum_lo(input, bias), constant),
                mb.cst[column],
            ),
            -penalty,
        );
        head_finite(lower)?;
        Ok(lower)
    }
}

// Diagnostic IDs are local to confirmation traces. Timing starts after tiled
// preparation and row-buffer allocation; it does not measure all of compose_viay.
pub(super) struct Trace(Option<ActiveTrace>);
struct ActiveTrace {
    id: u64,
    start: Instant,
    requested: usize,
    completed: usize,
    cached: usize,
    refused: usize,
    finished: bool,
}
impl Trace {
    pub(super) fn new(inputs: usize, head: usize, classes: usize) -> Self {
        if std::env::var("NY_MARGIN_ROW_ROOT_TIMING").is_err() {
            return Self(None);
        }
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let result = Self(Some(ActiveTrace {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            start: Instant::now(),
            requested: 0,
            completed: 0,
            cached: 0,
            refused: 0,
            finished: false,
        }));
        result.emit(
            "compose_begin",
            format_args!("inputs={inputs} head={head} classes={classes}"),
        );
        result
    }
    fn emit(&self, event: &str, details: std::fmt::Arguments<'_>) {
        if let Some(t) = &self.0 {
            use std::io::Write;
            let mut err = std::io::stderr().lock();
            let _ = writeln!(
                err,
                "[head-confirm] v=1 pid={} composition={} event={event} seconds={:.6} {details}",
                std::process::id(),
                t.id,
                t.start.elapsed().as_secs_f64()
            );
            let _ = err.flush();
        }
    }
    pub(super) fn event(&mut self, event: &str, column: usize, cached: bool) {
        if event == "row_begin" {
            if let Some(t) = &mut self.0 {
                t.requested += 1;
            }
        }
        self.emit(
            event,
            format_args!("column={column} cached={}", usize::from(cached)),
        );
    }
    pub(super) fn row_end(&mut self, column: usize, cached: bool, ok: bool) {
        if let Some(t) = &mut self.0 {
            t.completed += 1;
            t.cached += usize::from(cached);
            t.refused += usize::from(!ok);
        }
        self.emit(
            "row_end",
            format_args!(
                "column={column} cached={} accepted={}",
                usize::from(cached),
                usize::from(ok)
            ),
        );
    }
    pub(super) fn finish(&mut self) {
        if let Some(t) = &mut self.0 {
            t.finished = true;
        }
        self.summary("compose_end");
    }
    fn summary(&self, event: &str) {
        if let Some(t) = &self.0 {
            self.emit(
                event,
                format_args!(
                    "requested={} completed={} cached={} refused={}",
                    t.requested, t.completed, t.cached, t.refused
                ),
            );
        }
    }
}
impl Drop for Trace {
    fn drop(&mut self) {
        if self.0.as_ref().is_some_and(|t| !t.finished) {
            self.summary("compose_abort");
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static COUNTS: std::cell::Cell<[usize; 3]> = const { std::cell::Cell::new([0; 3]) };
}
#[cfg(test)]
fn observe(index: usize) {
    COUNTS.with(|counts| {
        let mut value = counts.get();
        value[index] += 1;
        counts.set(value);
    });
}
#[cfg(test)]
pub(super) fn with_counts<T>(run: impl FnOnce() -> T) -> (T, [usize; 3]) {
    struct Restore([usize; 3]);
    impl Drop for Restore {
        fn drop(&mut self) {
            COUNTS.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(COUNTS.with(|c| c.replace([0; 3])));
    let value = run();
    (value, COUNTS.with(|c| c.get()))
}

#[cfg(test)]
#[path = "bounds_confirm_tests.rs"]
mod tests;
