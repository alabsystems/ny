// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! G5: the minimum certified margin and a decade histogram (JS F8).
//!
//! The 272 banked tree-closed margin-row unsats that carry no recorded closing
//! margin are the argument for this line: a moat run that prints it makes the
//! "how close to zero did we close" audit decidable from the log alone.

/// Upper edges of the first nine decades; bucket `i` holds values in
/// `[EDGES[i-1], EDGES[i])`, bucket 0 everything below `1e-12`, bucket 9
/// everything at or above `1e-4`.
const EDGES: [f64; 9] = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4];

/// Buckets in a [`MarginHistogram`].
pub const DECADES: usize = EDGES.len() + 1;

/// Minimum certified margin plus a decade histogram of every close.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MarginHistogram {
    counts: [u64; DECADES],
    min: Option<f64>,
}

impl MarginHistogram {
    /// An empty histogram.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The bucket of a strictly positive finite margin.
    #[must_use]
    pub fn bucket(v: f64) -> usize {
        EDGES.iter().take_while(|&&edge| v >= edge).count()
    }

    /// Record one certified close. A value that is not finite and strictly
    /// positive is not a close and is ignored.
    pub fn note(&mut self, v: f64) {
        if !(v.is_finite() && v > 0.0) {
            return;
        }
        self.counts[Self::bucket(v)] += 1;
        self.min = Some(self.min.map_or(v, |m| m.min(v)));
    }

    /// Closes recorded.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// The smallest recorded margin.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.min
    }

    /// The bucket counts, smallest decade first.
    #[must_use]
    pub fn counts(&self) -> &[u64; DECADES] {
        &self.counts
    }

    /// The summary-line fragment:
    /// `closes=<n> min_margin=<e|none> margin_hist=<c0,…,c9>`.
    #[must_use]
    pub fn fragment(&self) -> String {
        let hist = self
            .counts
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let min = self
            .min
            .map_or_else(|| "none".to_string(), |m| format!("{m:.3e}"));
        format!(
            "closes={} min_margin={min} margin_hist={hist}",
            self.total()
        )
    }
}
