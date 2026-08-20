// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Admission: `Admitted(profile)` or `Declined(typed structural reason)`.
//!
//! # The one rule this module exists to make unbreakable
//!
//! Admission is a pure function of `(parsed spec, graph structure, remaining
//! budget)`. **The path, the category name and the preset key are not
//! arguments** — not because a convention says so, but because
//! [`AdmissionContext`] has no field that could carry one. There is nowhere to
//! put a filename. `tests/admission_is_structural.rs` re-reads this file and
//! fails if one appears.
//!
//! # Why the stage ladder is a type
//!
//! Evidence is evaluated cheapest-first: spec parse -> spec shape -> graph
//! structural scan -> model load. That ordering is the measured basis for
//! refusals costing 0.00-0.05 s, and ny already learned the cost of getting it
//! wrong: the exact-gradient scan "used to be discovered only INSIDE the
//! innermost step loop -- after printing the escalation banner, re-loading the
//! ONNX graph, re-reading the input shape, building the RNG/seed schedule and
//! running restart 0's trusted-ORT forward". A strategy declares its
//! [`AdmissionStage`] and the [`FactLadder`] is not descended past the deepest
//! stage any surviving strategy asked for.

use crate::proposal::StrategyName;
use core::time::Duration;

/// How deep into the evidence ladder a strategy's admission decision needs to
/// look. Ordered: cheaper stages sort first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdmissionStage {
    /// The VNN-LIB parses at all.
    SpecParse,
    /// The parsed spec has the right shape (box inputs, output atom kinds,
    /// free-dimension count, disjunct count).
    SpecShape,
    /// The ONNX graph's op set / terminal subgraph was scanned.
    GraphScan,
    /// The model was actually loaded and shape-inferred. The expensive one.
    ModelLoad,
}

/// Whether a lane may run at all. **Dark by default**, at the type level.
///
/// Defaults change only on a measurement, and this crate has no measurement
/// that would justify arming anything: E1 returned 0 counterexamples on 42
/// open-row measurements. A caller that wants these strategies on the scored
/// path must pass [`Arming::Armed`] explicitly, from behind a declared lever.
/// This crate reads no process environment — it cannot, it has no dependency
/// that would let it, and the lever ratchet forbids a raw read anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Arming {
    /// The shipped default.
    #[default]
    Dark,
    /// The caller has decided, on its own declared lever, to run this.
    Armed,
}

/// What kind of signal a strategy needs out of the objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveRequirement {
    /// Needs a true gradient through the network.
    Exact,
    /// Needs finite differences to be informative.
    EstimatedGradient,
    /// Needs only to compare two values. Survives a flat objective.
    ValueOnly,
}

/// Whether the objective carries usable search signal. A scheduler input, not
/// a log line (design §7.2): on a `Flat` objective every estimated-gradient
/// strategy is identically blind, and the value-only strategies must get the
/// budget instead. This is the mechanism by which `square` gets the seconds on
/// exactly the nets where it is the only thing that works.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveQuality {
    /// Distinct margins are observed at distinct points.
    Informative,
    /// Measured flat. On `traffic_signs` the max-other-minus-true margin is
    /// exactly `-1.0` at every sampled in-box point, integral, over exactly two
    /// distinct output values.
    Flat,
}

/// Worst-case time to yield. The scheduler sorts by this, and it must be the
/// worst case rather than the average: STE-first at its 240 s cap plus the LP
/// lane's measured worst case (131.1 s) plus the 45 s publication margin plus
/// the 100 s downstream reserve is 517 s against 456 s available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CostClass {
    /// Yields in a fixed, tiny number of oracle calls regardless of budget.
    Instant,
    /// Yields within a declared bound.
    Bounded(Duration),
    /// Will use every second it is given.
    Openended,
}

/// What a bigger budget is allowed to widen. A larger cap must change the
/// *trajectory*, not merely the endpoint: on cifar100, 20 open rows at official
/// budget converted 0, the same 20 at 2x BaB work converted 0, and 10 rows at
/// 4x budget converted 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamSpace {
    /// Free-dimension ceiling. **Per strategy, never shared** — letting the LP
    /// lane inherit the STE lane's 32768 free-unit cap took
    /// `model_30_idx_1703_eps_15` from 1.3 s to 139.4 s for 0 accepted flips.
    pub free_dims_ceiling: usize,
    /// Hard cap on points this strategy may construct.
    pub max_points: usize,
    /// How many independent restarts a wider budget may buy.
    pub max_restarts: usize,
}

/// Structural facts read off the parsed spec. Stages `SpecParse`/`SpecShape`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecFacts {
    /// Input coordinates with `hi > lo`.
    pub free_dims: usize,
    /// Input coordinates pinned to a constant by an equality.
    pub pinned_dims: usize,
    /// The shape of the property.
    pub shape: SpecShape,
    /// Disjuncts of a top-level `or`, each of which is a steering target.
    pub disjunct_targets: usize,
    /// The spec contains equality atoms on outputs.
    pub has_equality_atoms: bool,
}

/// The property shapes admission distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecShape {
    /// Box input constraints plus an output condition. The ordinary case.
    BoxInputs,
    /// A low-dimensional top-level relational conjunction; its optimum can sit
    /// on a face interior, which corner enumeration misses outright.
    LowDimRelationalConjunction,
    /// Input assertions that are not a box. Nothing here can search it.
    NonBoxInputAssertions {
        /// How many assertions could not be reduced to a per-coordinate bound.
        non_box_assertions: usize,
    },
}

/// Structural facts read off the ONNX graph. Stage `GraphScan`. Neither ported
/// strategy needs this, which is exactly why both refuse in microseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFacts {
    /// First op that is outside the exact-gradient fragment, if any.
    pub first_unsupported_gradient_op: Option<String>,
    /// The terminal op saturates in f32 (Softmax over large logit gaps).
    pub terminal_saturates: bool,
}

/// Why a strategy was not admitted. **Never an `Option::None`** — a decline is
/// a receipt, and the receipt is what licenses widening the strategy to a new
/// family later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decline {
    /// Outside the exact-gradient fragment.
    OutsideExactGradientFragment {
        /// The op that ended the scan.
        first_unsupported_op: String,
    },
    /// Too many free input dimensions for this strategy's own ceiling.
    FreeDimsAboveCeiling {
        /// Free dimensions in the spec.
        free: usize,
        /// This strategy's ceiling.
        ceiling: usize,
    },
    /// Too few free input dimensions for this strategy to do anything.
    FreeDimsBelowFloor {
        /// Free dimensions in the spec.
        free: usize,
        /// This strategy's floor.
        floor: usize,
    },
    /// The property is not a shape this strategy can search.
    SpecShapeUnsupported {
        /// What the strategy needs.
        want: SpecShape,
        /// What the spec is.
        got: SpecShape,
    },
    /// A fragment-specific refusal owned by another crate, rendered as text so
    /// this crate does not have to depend on that crate to carry it.
    FragmentRefusal {
        /// Which fragment refused.
        fragment: String,
        /// The refusing crate's own typed reason, rendered.
        reason: String,
    },
    /// Not enough budget left to be worth entering.
    BudgetBelowFloor {
        /// What is left.
        remaining: Duration,
        /// What this strategy needs at minimum.
        floor: Duration,
    },
    /// The objective cannot supply the signal this strategy requires. This is
    /// the `#deadlane` refusal, made typed: an estimated-gradient strategy on a
    /// flat objective is not "slow", it is blind.
    ObjectiveTooWeak {
        /// What the strategy needs.
        want: ObjectiveRequirement,
        /// What the objective supplies.
        got: ObjectiveQuality,
    },
    /// Dark by default. Not a structural fact about the problem.
    Disarmed,
}

impl Decline {
    /// True when the decline is a fact about the *problem* rather than about
    /// configuration. Only a structural decline licenses a claim like "this
    /// family is out of reach"; `Disarmed` licenses nothing.
    pub const fn is_structural(&self) -> bool {
        !matches!(self, Self::Disarmed)
    }
}

/// Admitted, with the reach/cost class the scheduler needs to allocate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionProfile {
    /// What the strategy needs from the objective.
    pub objective: ObjectiveRequirement,
    /// Needs a near-miss incumbent to refine (S12, S13).
    pub needs_incumbent: bool,
    /// Needs BaB frontier seeds (S15).
    pub needs_verifier_state: bool,
    /// Free input dimensions it will search.
    pub free_dims: usize,
    /// Worst-case yield time.
    pub declared_cost: CostClass,
    /// What a bigger budget may widen.
    pub params: ParamSpace,
}

/// The admission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// Admitted with a profile.
    Admitted(AdmissionProfile),
    /// Declined with a typed structural reason. Never a silence.
    Declined(Decline),
}

/// Everything a strategy is allowed to look at when deciding admission.
///
/// Note what is absent: there is no path, no filename, no directory, no
/// category string, no preset key and no instance identifier. Presets configure
/// BUDGET and ARMING; they may not say "this network is a BNN".
#[derive(Debug, Clone, Copy)]
pub struct AdmissionContext<'a> {
    spec: &'a SpecFacts,
    graph: Option<&'a GraphFacts>,
    remaining: Duration,
    arming: Arming,
    objective: ObjectiveQuality,
}

impl<'a> AdmissionContext<'a> {
    /// Build a context from spec evidence alone (stages `SpecParse`/`SpecShape`).
    pub const fn new(
        spec: &'a SpecFacts,
        remaining: Duration,
        arming: Arming,
        objective: ObjectiveQuality,
    ) -> Self {
        Self {
            spec,
            graph: None,
            remaining,
            arming,
            objective,
        }
    }

    /// Attach graph evidence (stage `GraphScan`). Only called when a strategy
    /// that survived the cheaper stages actually asked for it.
    #[must_use]
    pub const fn with_graph(mut self, graph: &'a GraphFacts) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Parsed-spec evidence.
    pub const fn spec(&self) -> &SpecFacts {
        self.spec
    }

    /// Graph evidence, if the ladder has been descended that far.
    pub const fn graph(&self) -> Option<&GraphFacts> {
        self.graph
    }

    /// Budget remaining for the whole falsification phase.
    pub const fn remaining(&self) -> Duration {
        self.remaining
    }

    /// Whether the caller has armed this lane.
    pub const fn arming(&self) -> Arming {
        self.arming
    }

    /// Measured objective quality.
    pub const fn objective(&self) -> ObjectiveQuality {
        self.objective
    }
}

/// The lazily-descended evidence ladder.
///
/// Implemented by the caller (in ny, by `ny-cli`, over its own parser and ONNX
/// loader). The registry calls these in order and never calls a deeper one than
/// some surviving strategy needs, which is what keeps a refusal at 0.00-0.05 s.
pub trait FactLadder {
    /// Stage 1+2. Parse the spec and read its shape.
    fn spec_facts(&mut self) -> Result<SpecFacts, Decline>;
    /// Stage 3. Scan the graph structurally. Not called unless needed.
    fn graph_facts(&mut self) -> Result<GraphFacts, Decline>;
    /// Stage 4. Load and shape-infer the model. Not called unless needed.
    fn load_model(&mut self) -> Result<(), Decline>;
}

/// A per-strategy admission receipt, with the elapsed time that licenses the
/// claim that the refusal was free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionReceipt {
    /// Which strategy.
    pub strategy: StrategyName,
    /// The decision.
    pub admission: Admission,
    /// How long the decision took.
    pub elapsed: Duration,
    /// The deepest ladder stage that was actually descended for it.
    pub deepest_stage_reached: AdmissionStage,
}
