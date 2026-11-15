// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded, untrusted convex ratio proposal. Every accepted relation still
//! checks all original coordinates and then uses the old original evaluator.
use std::time::Instant;

// Four phase searches, <=32 breakpoints each, stack sort, derivative scans
// and <=32-coordinate scoring. Separate from the complete original replay.
pub(super) const WORK: u64 = 4096;

pub(super) fn propose(
    a: &[f64],
    b: &[f64],
    center: [f64; 2],
    radius: [f64; 2],
    original_columns: usize,
    deadline: Option<Instant>,
) -> Option<f64> {
    if a.len() != b.len()
        || a.is_empty()
        || a.len() > super::SAMPLE
        || original_columns < a.len()
        || a.iter()
            .chain(b)
            .chain(&center)
            .chain(&radius)
            .any(|v| !v.is_finite())
        || radius.iter().any(|v| *v < 0.0)
        || super::super::past(deadline)
    {
        return None;
    }
    // Sample the original objective on a consistent scale. Neither this
    // estimate nor the rounded cancellation point is certificate authority.
    let q_scale = a.len() as f64 / original_columns as f64;
    let mut best: Option<(f64, f64)> = None;
    for s0 in [-1.0, 1.0] {
        for s1 in [-1.0, 1.0] {
            if super::super::past(deadline) {
                return None;
            }
            let sign = s0 * s1;
            let q0 = (s0 * center[0] + radius[0]) * q_scale;
            let q1 = (s1 * center[1] + radius[1]) * q_scale;
            let mut derivative = q0 - q1;
            let mut points = [(0.0, 0.0); super::SAMPLE];
            let mut count = 0;
            for (k, (&a, &b)) in a.iter().zip(b).enumerate() {
                if k % 16 == 0 && super::super::past(deadline) {
                    return None;
                }
                let e = sign * b;
                let d = a - e;
                if !d.is_finite() {
                    return None;
                }
                derivative += if e == 0.0 { d.abs() } else { d * e.signum() };
                if d != 0.0 {
                    let t = -e / d;
                    if t > 0.0 && t < 1.0 {
                        points[count] = (t, 2.0 * d.abs());
                        count += 1;
                    }
                }
            }
            if !derivative.is_finite() || derivative >= 0.0 {
                continue;
            }
            points[..count].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let mut candidate = None;
            for &(t, jump) in &points[..count] {
                derivative += jump;
                if derivative >= 0.0 {
                    candidate = Some(t);
                    break;
                }
            }
            let Some(w) = candidate else {
                continue;
            };
            let mut score = w * q0 + (1.0 - w) * q1;
            for (&a, &b) in a.iter().zip(b) {
                score += (w * a + sign * (1.0 - w) * b).abs();
            }
            if score.is_finite() && score < 0.0 && best.is_none_or(|(old, _)| score < old) {
                best = Some((score, w));
            }
        }
    }
    if super::super::past(deadline) {
        None
    } else {
        best.map(|(_, w)| w)
    }
}
