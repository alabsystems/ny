// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared test doubles. Included with `#[path]`, not a test binary of its own.

#![allow(dead_code)]

use ny_falsify::{
    Decline, FactLadder, GraphFacts, Oracle, OracleError, Score, SpecFacts, SpecShape,
};

/// A `FactLadder` that counts how deep anything actually descended.
///
/// The counters are the whole point: "admission is cheap" is not a claim about
/// a stopwatch, it is a claim that the expensive stages were NOT ENTERED, and
/// a stopwatch on a fast machine cannot distinguish those.
pub struct CountingLadder {
    pub spec: SpecFacts,
    pub graph: GraphFacts,
    pub spec_calls: usize,
    pub graph_calls: usize,
    pub model_calls: usize,
}

impl CountingLadder {
    pub fn new(spec: SpecFacts) -> Self {
        Self {
            spec,
            graph: GraphFacts {
                first_unsupported_gradient_op: None,
                terminal_saturates: false,
            },
            spec_calls: 0,
            graph_calls: 0,
            model_calls: 0,
        }
    }
}

impl FactLadder for CountingLadder {
    fn spec_facts(&mut self) -> Result<SpecFacts, Decline> {
        self.spec_calls += 1;
        Ok(self.spec.clone())
    }
    fn graph_facts(&mut self) -> Result<GraphFacts, Decline> {
        self.graph_calls += 1;
        Ok(self.graph.clone())
    }
    fn load_model(&mut self) -> Result<(), Decline> {
        self.model_calls += 1;
        Ok(())
    }
}

/// Ordinary box spec facts with `free` free dimensions.
pub fn box_spec(free: usize) -> SpecFacts {
    SpecFacts {
        free_dims: free,
        pinned_dims: 0,
        shape: SpecShape::BoxInputs,
        disjunct_targets: 1,
        has_equality_atoms: false,
    }
}

/// An oracle driven by a caller-supplied predicate over the full input vector.
///
/// `steer` is what the search hill-climbs; `holds` is the caller's own
/// arithmetic about its own point. Note that the oracle -- not the strategy --
/// is what decides `holds`, which is the seam that keeps a verdict out of this
/// crate entirely.
pub struct PredicateOracle<F: FnMut(&[f64]) -> Score> {
    pub score: F,
    pub calls: usize,
    pub points: usize,
    pub batch: usize,
}

impl<F: FnMut(&[f64]) -> Score> PredicateOracle<F> {
    pub fn new(score: F) -> Self {
        Self {
            score,
            calls: 0,
            points: 0,
            batch: 256,
        }
    }
}

impl<F: FnMut(&[f64]) -> Score> Oracle for PredicateOracle<F> {
    fn evaluate_batch(&mut self, points: &[Vec<f64>]) -> Result<Vec<Score>, OracleError> {
        self.calls += 1;
        self.points += points.len();
        Ok(points.iter().map(|p| (self.score)(p)).collect())
    }
    fn batch_limit(&self) -> usize {
        self.batch
    }
}
