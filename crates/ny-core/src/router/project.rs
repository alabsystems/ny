// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PURE progress projections (ROUTER_PROGRAM §4.3): FAR / NOT-FAR per progress
//! kind, computed in WORK UNITS from the trailing windows of one capability.
//!
//! There is no NEAR class: at the first closure observation no unsat record
//! exceeds closes/expansion 1.0, and 6 of 8 first cross it at their LAST
//! window, so "near" is not observable (§1.1). A window above break-even is
//! noise. Every function here is f64 or saturating integer arithmetic; a zero
//! denominator is `None` / [`FarClass::Unknown`], never an infinity cast into
//! an integer. No clock, no environment, no filesystem.

use serde::{Deserialize, Serialize};

use super::bus::ProgressSample;
use super::decl::ProgressKind;

// FITTED on one family (judge_ig_far_rule.out); pre-registered in R0.
// The most conservative value of the tested range [0.10, 0.30]: 0/8 unsat
// records and 26/41 timeouts flagged on cifar100/tiny/cct margin-row logs.
// It must pass the full-corpus replay (false-FAR = 0 on every eventually
// decided record) before any acting use.
/// Closes-per-expansion ceiling of a FAR window.
pub const T_FAR: f64 = 0.10;

/// Work units per frontier window: the shipped `#closure-adaptive` window
/// (16 expansions, ~32 children), the window the FAR evidence was read in.
pub const FRONTIER_WINDOW_UNITS: u64 = 16;

/// A BFS class is classifiable only past this depth (§4.3 "two levels past
/// depth 8").
pub const BFS_FAR_MIN_DEPTH: u64 = 8;

/// A certified-volume window is flat when its gain is at most this.
pub const VOLUME_FLAT_EPS: f64 = 1e-3;

/// Default attack-margin noise floor when the lane supplies none.
pub const MARGIN_NOISE_EPS: f64 = 1e-3;

/// A margin trace is classifiable only when it spans at least this fraction
/// of the attack slice (`attack_stall.rs`, W >= 0.5 of the slice).
pub const MARGIN_MIN_SPAN_FRAC: f64 = 0.5;

/// `rule.governor` yield ratio: a phase yields when its per-iteration gain
/// falls below this fraction of its own best gain. Selected on cifar100:
/// detector(n=1) until a second family reproduces a converting stop.
pub const GOVERNOR_RATIO: f64 = 0.15;

/// Samples the checkpoint keeps per capability (two windows).
pub const TRAILING_SAMPLES: usize = 3;

/// A FAR / NOT-FAR classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FarClass {
    /// Two trailing windows say the capability will not finish.
    Far,
    /// Not FAR (which is NOT a claim that it is near).
    NotFar,
    /// No evidence: too few windows, a zero denominator, or a kind that is
    /// never classified (opaque, dual, yield).
    Unknown,
}

impl FarClass {
    /// Stable token for logs.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Far => "far",
            Self::NotFar => "not_far",
            Self::Unknown => "n/a",
        }
    }
}

/// The delta between two samples of one capability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    /// Work done in the window.
    pub work: u64,
    /// Closes in the window.
    pub closes: u64,
    /// Prunes in the window.
    pub pruned: u64,
    /// Open frontier at the window's start.
    pub open_before: u64,
    /// Open frontier at the window's end.
    pub open_after: u64,
    /// Level / depth at the window's end.
    pub level: u64,
    /// Scalar value at the window's start.
    pub value_before: f64,
    /// Scalar value at the window's end.
    pub value_after: f64,
    /// Window start, seconds.
    pub t0: f64,
    /// Window end, seconds.
    pub t1: f64,
}

impl Window {
    /// Whether the open frontier grew across the window.
    #[must_use]
    pub const fn open_grew(&self) -> bool {
        self.open_after > self.open_before
    }
}

/// The window from `a` to `b` (saturating: a counter that went backwards is a
/// zero-work window, which every projection reads as `Unknown`).
#[must_use]
pub fn window(a: &ProgressSample, b: &ProgressSample) -> Window {
    Window {
        work: b.work_done.saturating_sub(a.work_done),
        closes: b.closes.saturating_sub(a.closes),
        pruned: b.pruned.saturating_sub(a.pruned),
        open_before: a.open,
        open_after: b.open,
        level: b.level,
        value_before: a.margin_best(),
        value_after: b.margin_best(),
        t0: a.secs(),
        t1: b.secs(),
    }
}

/// A window's rates. `closes_per_unit` DECIDES; `units_per_sec` only judges
/// affordability (§2.4: work ratios agreed 12/12 across reps, wall rates
/// differed by up to 35%).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// Closes per unit of work (`None` on zero work).
    pub closes_per_unit: Option<f64>,
    /// Units of work per second (`None` on a zero or non-finite span).
    pub units_per_sec: Option<f64>,
}

/// The rates of one window.
#[must_use]
pub fn window_rates(w: &Window) -> Rates {
    let closes_per_unit = (w.work > 0).then(|| w.closes as f64 / w.work as f64);
    let span = w.t1 - w.t0;
    let units_per_sec = (span.is_finite() && span > 0.0).then(|| w.work as f64 / span);
    Rates {
        closes_per_unit,
        units_per_sec,
    }
}

/// The trailing samples of capability `cap` at or before `t_secs`: at most
/// [`TRAILING_SAMPLES`], oldest first, from the LAST monotone run of its work
/// counter, each at least `window_units` of work after the previous one.
///
/// Idempotent: selecting from the selection returns the selection, which is
/// what lets a flight record carry only these samples and still replay.
#[must_use]
pub fn trailing(
    samples: &[ProgressSample],
    cap: &str,
    t_secs: f64,
    window_units: u64,
) -> Vec<ProgressSample> {
    let mut mine: Vec<ProgressSample> = samples
        .iter()
        .filter(|s| s.cap.as_str() == cap && s.secs() <= t_secs)
        .copied()
        .collect();
    mine.sort_by(|a, b| a.secs().total_cmp(&b.secs()));
    let Some(last) = mine.last().copied() else {
        return Vec::new();
    };
    // The trailing monotone run: stop at the first counter reset.
    let mut start = mine.len() - 1;
    while start > 0 && mine[start - 1].work_done <= mine[start].work_done {
        start -= 1;
    }
    let run = &mine[start..];
    let step = window_units.max(1);
    let mut picked = vec![last];
    let mut bound = last.work_done;
    for sample in run.iter().rev().skip(1) {
        if picked.len() == TRAILING_SAMPLES {
            break;
        }
        let Some(limit) = bound.checked_sub(step) else {
            break;
        };
        if sample.work_done <= limit {
            picked.push(*sample);
            bound = sample.work_done;
        }
    }
    picked.reverse();
    picked
}

/// FrontierBinary / FrontierKary: FAR iff closes/unit <= `t_far` in BOTH
/// windows and the open frontier grew over both. The identity it rests on:
/// `created = 2 exp + 1`, so `Δopen = exp − closes` and break-even is 1.0.
#[must_use]
pub fn far_frontier_binary(w1: &Window, w2: &Window, t_far: f64) -> FarClass {
    match (
        window_rates(w1).closes_per_unit,
        window_rates(w2).closes_per_unit,
    ) {
        (Some(r1), Some(r2)) => {
            if r1 <= t_far && r2 <= t_far && w1.open_grew() && w2.open_grew() {
                FarClass::Far
            } else {
                FarClass::NotFar
            }
        }
        _ => FarClass::Unknown,
    }
}

/// The live open frontier of a level-synchronous BFS from two counters the
/// tier already keeps: `open = N − 2P + 1`, where `N` counts every evaluated
/// node (root included) and `P` the prunes. A closed tree reads 0. `None`
/// when the counters are inconsistent (`2P > N + 1`) or overflow.
#[must_use]
pub fn bfs_open(n: u64, p: u64) -> Option<u64> {
    n.checked_add(1)?.checked_sub(p.checked_mul(2)?)
}

/// A level's growth `g = 2 (1 − pruned / evaluated)`: the next level holds
/// `g` times this one. `None` on an empty or inconsistent level.
#[must_use]
pub fn bfs_growth(evaluated: u64, pruned: u64) -> Option<f64> {
    if evaluated == 0 || pruned > evaluated {
        return None;
    }
    Some(2.0 * (1.0 - pruned as f64 / evaluated as f64))
}

/// Bfs: FAR iff both trailing levels are past [`BFS_FAR_MIN_DEPTH`] and grow
/// (`g >= 1`).
#[must_use]
pub fn far_bfs(w1: &Window, w2: &Window) -> FarClass {
    match (
        bfs_growth(w1.work, w1.pruned),
        bfs_growth(w2.work, w2.pruned),
    ) {
        (Some(g1), Some(g2)) => {
            if w1.level > BFS_FAR_MIN_DEPTH
                && w2.level > BFS_FAR_MIN_DEPTH
                && g1 >= 1.0
                && g2 >= 1.0
            {
                FarClass::Far
            } else {
                FarClass::NotFar
            }
        }
        _ => FarClass::Unknown,
    }
}

/// Volume: FAR iff the certified volume fraction is flat (gain <= `eps`) and
/// the queue grew, in both windows.
#[must_use]
pub fn far_volume(w1: &Window, w2: &Window, eps: f64) -> FarClass {
    let gain = |w: &Window| {
        let g = w.value_after - w.value_before;
        g.is_finite().then_some(g)
    };
    match (gain(w1), gain(w2)) {
        (Some(g1), Some(g2)) if eps.is_finite() => {
            if g1 <= eps && g2 <= eps && w1.open_grew() && w2.open_grew() {
                FarClass::Far
            } else {
                FarClass::NotFar
            }
        }
        _ => FarClass::Unknown,
    }
}

/// Least-squares slope of `(secs, margin)` points; `None` with fewer than two
/// finite points or no time spread.
#[must_use]
pub fn margin_slope(points: &[(f64, f64)]) -> Option<f64> {
    let finite: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(t, m)| t.is_finite() && m.is_finite())
        .collect();
    if finite.len() < 2 {
        return None;
    }
    let n = finite.len() as f64;
    let mean_t = finite.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_m = finite.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0f64, 0.0f64);
    for (t, m) in &finite {
        sxy += (t - mean_t) * (m - mean_m);
        sxx += (t - mean_t) * (t - mean_t);
    }
    let slope = sxy / sxx;
    (sxx > 0.0 && slope.is_finite()).then_some(slope)
}

/// Margin: FAR iff the best-margin RATE is within noise over a span of at
/// least half the slice: `slope * W <= eps`. The LEVEL never decides (a
/// −0.033 flat trace is FAR; a −0.083 climbing one is not).
#[must_use]
pub fn far_margin(points: &[(f64, f64)], slice_secs: f64, eps: f64) -> FarClass {
    let finite_t = points.iter().map(|p| p.0).filter(|t| t.is_finite());
    let (lo, hi) = finite_t.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
        (lo.min(t), hi.max(t))
    });
    let span = hi - lo;
    if !(span.is_finite() && slice_secs.is_finite() && eps.is_finite())
        || span < MARGIN_MIN_SPAN_FRAC * slice_secs
        || span <= 0.0
    {
        return FarClass::Unknown;
    }
    match margin_slope(points) {
        Some(slope) if slope * span <= eps => FarClass::Far,
        Some(_) => FarClass::NotFar,
        None => FarClass::Unknown,
    }
}

/// Screen: FAR iff `roots_per_sec * remaining < roots_needed / 10`.
#[must_use]
pub fn far_screen(roots_per_sec: f64, remaining_secs: f64, roots_needed: f64) -> FarClass {
    let reach = roots_per_sec * remaining_secs;
    if !(reach.is_finite() && roots_needed.is_finite()) || roots_per_sec < 0.0 {
        return FarClass::Unknown;
    }
    if reach < roots_needed / 10.0 {
        FarClass::Far
    } else {
        FarClass::NotFar
    }
}

/// `rule.governor` (O-STOP, shadow in R1): the first iteration whose gain falls
/// below `ratio` of the phase's own best gain, once at least `min_samples`
/// finite gains were seen. `None` = the rule would not stop the phase.
#[must_use]
pub fn governor_yield_index(gains: &[f64], ratio: f64, min_samples: usize) -> Option<usize> {
    if !ratio.is_finite() || ratio <= 0.0 {
        return None;
    }
    let mut best = 0.0f64;
    let mut seen = 0usize;
    for (index, &gain) in gains.iter().enumerate() {
        if !gain.is_finite() {
            continue;
        }
        seen += 1;
        if seen > min_samples && best > 0.0 && gain < ratio * best {
            return Some(index);
        }
        best = best.max(gain);
    }
    None
}

/// Classify one capability from its trailing samples (oldest first).
#[must_use]
pub fn classify(kind: ProgressKind, trailing: &[ProgressSample]) -> FarClass {
    let windows = |s: &[ProgressSample]| match s {
        [a, b, c] => Some((window(a, b), window(b, c))),
        _ => None,
    };
    match kind {
        ProgressKind::FrontierBinary | ProgressKind::FrontierKary => windows(trailing)
            .map_or(FarClass::Unknown, |(w1, w2)| {
                far_frontier_binary(&w1, &w2, T_FAR)
            }),
        ProgressKind::Bfs => {
            windows(trailing).map_or(FarClass::Unknown, |(w1, w2)| far_bfs(&w1, &w2))
        }
        ProgressKind::Volume => windows(trailing).map_or(FarClass::Unknown, |(w1, w2)| {
            far_volume(&w1, &w2, VOLUME_FLAT_EPS)
        }),
        ProgressKind::Margin => {
            // The router does not know an attack's slice in R1: the trace's own
            // span is the slice, so only the rate can decide.
            let points: Vec<(f64, f64)> = trailing
                .iter()
                .map(|s| (s.secs(), s.margin_best()))
                .collect();
            let span = points
                .first()
                .zip(points.last())
                .map_or(0.0, |(a, b)| b.0 - a.0);
            far_margin(&points, span, MARGIN_NOISE_EPS)
        }
        ProgressKind::Screen | ProgressKind::Opaque | ProgressKind::Dual | ProgressKind::Yield => {
            FarClass::Unknown
        }
    }
}

/// The work spacing of one progress kind's windows.
#[must_use]
pub const fn window_units(kind: ProgressKind) -> u64 {
    match kind {
        ProgressKind::FrontierBinary | ProgressKind::FrontierKary => FRONTIER_WINDOW_UNITS,
        _ => 1,
    }
}
