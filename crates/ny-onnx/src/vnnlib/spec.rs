// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// A single output constraint (relational property).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum OutputConstraint {
    /// Y_i <= Y_j
    LessEq(usize, usize),
    /// Y_i >= Y_j (equivalent to Y_j <= Y_i)
    GreaterEq(usize, usize),
    /// Y_i < Y_j
    LessThan(usize, usize),
    /// Y_i > Y_j
    GreaterThan(usize, usize),
    /// Y_i <= constant
    LessEqConst(usize, f64),
    /// Y_i >= constant
    GreaterEqConst(usize, f64),
    /// Y_i < constant
    LessThanConst(usize, f64),
    /// Y_i > constant
    GreaterThanConst(usize, f64),
}

impl OutputConstraint {
    /// Whether this comparison is strict (`<`/`>`, unsatisfied at equality).
    /// Non-strict constraints (`<=`/`>=`) are satisfied at exact equality —
    /// relevant for SAT-encoded benchmarks (sat_relu) whose satisfying
    /// assignments land outputs EXACTLY on the threshold (dyadic-exact
    /// arithmetic), making margin 0.0 the maximum achievable.
    pub fn is_strict(&self) -> bool {
        matches!(
            self,
            Self::LessThan(..)
                | Self::GreaterThan(..)
                | Self::LessThanConst(..)
                | Self::GreaterThanConst(..)
        )
    }
}

/// VNN-LIB 2.0 relation between two declared networks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkRelation {
    /// The second network is structurally isomorphic to the first; the property
    /// is epsilon-equivalence over equal inputs.
    IsomorphicTo,
    /// The two declarations refer to the same network used at related inputs.
    EqualTo,
    /// The declaring "network" is a symbolic geometric ground truth: its
    /// semantics come from a `.gt.json` sidecar (see `ny-groundtruth`), named
    /// by this path (or inline reference) exactly as written in the file —
    /// `(ground-truth "cyl.gt.json")`. Resolution through the sidecar loader
    /// happens at the verify layer, which also resolves relative paths against
    /// the VNN-LIB file's directory.
    GroundTruth(String),
}

/// A declared network in a VNN-LIB 2.0 multi-network property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredNetwork {
    pub name: String,
    pub input: String,
    pub output: String,
    /// Declared VNN-LIB element type for the input tensor (for example
    /// `float32` or `real`). Counterexample serialization must reproduce this
    /// type in the VNN-LIB 2.0 textual assignment header.
    pub input_type: String,
    /// Declared VNN-LIB element type for the output tensor.
    pub output_type: String,
    /// Declared input shape, before row-major flattening.
    pub input_shape: Vec<usize>,
    /// Declared output shape, before row-major flattening.
    pub output_shape: Vec<usize>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub relation_to: Option<(NetworkRelation, String)>,
}

/// Kind of a tensor definition in the mandatory VNN-LIB 2.0 textual
/// counterexample assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDeclarationKind {
    Input,
    Hidden,
    Output,
}

/// Exact tensor declaration metadata needed to serialize a VNN-LIB 2.0
/// counterexample. Declarations are returned in the reference checker's order:
/// networks in source order, then inputs, hidden tensors, and outputs within
/// each network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDeclaration {
    pub network: Option<String>,
    pub name: String,
    pub element_type: String,
    pub shape: Vec<usize>,
    pub kind: TensorDeclarationKind,
}

/// A parsed dual-network property for SOUND relational verification.
#[derive(Debug, Clone, PartialEq)]
pub enum DualNetworkProperty {
    /// Prove |f_i(x) - g_i(x)| <= epsilon for every output i.
    EpsilonEquivalence { epsilon: f64 },
    /// Prove the safe complement for the monotonic ACAS relation. If the unsafe
    /// VNN-LIB clause is strict (`f < g`), proving `f - g >= 0` is sufficient.
    /// If the unsafe clause is non-strict (`f <= g`), equality remains unsafe and
    /// the verifier must prove a strictly positive margin.
    MonotonicGreaterEq {
        output: usize,
        varying_input: usize,
        strict_unsafe: bool,
    },
    /// Prove the first network dominates the second over the shared input:
    /// `h(x) = f(x) − g(x) ≥ 0` for every output (the ground-truth dominance
    /// property; the second "network" is typically a
    /// [`NetworkRelation::GroundTruth`] graph). The unsafe VNN-LIB clause is
    /// `f < g` (strict, `strict_unsafe == true`, `h ≥ 0` suffices) or
    /// `f <= g` (non-strict, equality remains unsafe and the verifier must
    /// prove a strictly positive margin).
    DominatesSecond { strict_unsafe: bool },
}

/// Comparison relation for a parsed isomorphic output deviation atom.
///
/// These mirror the four strict/non-strict VNN-LIB relations relevant to the
/// difference `t = Y_g[i] - Y_f[i]`. The relation is recorded EXACTLY as parsed
/// (no `.abs()` normalization of the constant), so a downstream Farkas check
/// operates on the real signed region rather than a hard-coded template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsomorphicAtomRelation {
    /// `t > c`
    Gt,
    /// `t < c`
    Lt,
    /// `t >= c`
    Ge,
    /// `t <= c`
    Le,
}

/// A single parsed isomorphic output deviation atom, in canonical
/// difference form `t ⋈ c` where `t = Y_g[index] - Y_f[index]` and `c` is the
/// REAL signed constant (numerator/denominator of its exact dyadic value).
///
/// The constant is stored as a signed `f64` exactly as it appears in the
/// VNN-LIB source (no `.abs()`); the consuming Farkas gate is responsible for
/// converting it to an exact rational and proving infeasibility of the real
/// region. Storing the true signed value is what severs the certificate from a
/// hard-coded `±eps` template.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsomorphicOutputAtom {
    /// Output index `i` the atom constrains.
    pub index: usize,
    /// The relation in canonical `t ⋈ c` form.
    pub relation: IsomorphicAtomRelation,
    /// The REAL signed right-hand-side constant `c` (NOT `.abs()`-normalized).
    pub constant: f64,
}

/// Structural facts extracted from the original VNN-LIB dual-network formula.
///
/// These are deliberately facts, not interpretations: the VNN-COMP relational
/// shortcut may use them to decide whether a difference-network proof exactly
/// matches the parsed property, and must fall back to `unknown` otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct DualNetworkValidation {
    /// `input_equalities[i]` is true only when the VNN-LIB explicitly asserted
    /// `f_input[i] == g_input[i]` (in either syntactic order).
    pub input_equalities: Vec<bool>,
    /// `f_input_ge_g_input[i]` is true only when the VNN-LIB explicitly asserted
    /// `f_input[i] >= g_input[i]` or a stricter equivalent order.
    pub f_input_ge_g_input: Vec<bool>,
    /// `g_input_ge_f_input[i]` is true only when the VNN-LIB explicitly asserted
    /// `g_input[i] >= f_input[i]` or a stricter equivalent order.
    pub g_input_ge_f_input: Vec<bool>,
    /// True only for the canonical epsilon-equivalence unsafe complement shape
    /// that has same-index positive and negative deviations for every output,
    /// with one shared strict epsilon. Structure-agnostic: the atoms may be
    /// combined disjunctively (the canonical complement IS an or-of-ors);
    /// consumers that need a purely conjunctive region (the Farkas emptiness
    /// shortcut) must additionally check `isomorphic_output_is_conjunction`.
    pub isomorphic_output_safe_complement: bool,
    /// Number of canonical same-index monotonic unsafe output comparisons found.
    pub monotonic_output_relation_count: usize,
    /// True when an f/g output comparison was seen but did not match the
    /// canonical same-index relation this parser can validate.
    pub unsupported_output_relation: bool,
    /// The REAL parsed isomorphic output deviation atoms, in canonical
    /// difference form `t = Y_g[i] - Y_f[i] ⋈ c` with the true signed constant
    /// `c` preserved. The Farkas emptiness gate builds its certificate from
    /// THESE atoms (not a `±eps` template), so a passing certificate is a
    /// genuine proof that the real unsafe region is empty.
    pub isomorphic_output_atoms: Vec<IsomorphicOutputAtom>,
    /// True only when every parsed isomorphic output deviation atom occurs in a
    /// purely CONJUNCTIVE position (no enclosing `or`). When any output atom
    /// appears under an `or`, the unsafe region is a disjunction `|t| > eps`
    /// (feasible for distinct f/g), the property does NOT universally hold, and
    /// the emptiness shortcut MUST decline. Defaults to `true` (vacuously, when
    /// there are no output atoms) and is cleared on the first `or`-enclosed
    /// output atom encountered.
    pub isomorphic_output_is_conjunction: bool,
}

/// Parsed VNN-LIB 2.0 dual-network metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DualNetworkSpec {
    pub networks: Vec<DeclaredNetwork>,
    pub property: DualNetworkProperty,
    /// True only when the VNN-LIB explicitly asserts `f_input[i] == g_input[i]`
    /// for every input index and the parsed f/g input boxes match exactly. The
    /// isomorphic shared-input difference-network verifier may emit `unsat`
    /// only under this condition; otherwise f and g may range over independent
    /// inputs and the conservative result is `unknown`.
    pub shared_input_coupling: bool,
    /// Bounds for the original f input variables.
    pub f_input_bounds: Vec<(f64, f64)>,
    /// Bounds for the derived g/base input variables. For epsilon-equivalence
    /// this is collapsed to `f_input_bounds` only when `shared_input_coupling`
    /// is true.
    pub g_input_bounds: Vec<(f64, f64)>,
    /// Explicit structural validation facts from the parsed VNN-LIB.
    pub validation: DualNetworkValidation,
    /// FULL-COVERAGE DNF of the asserted formula over exact-rational linear
    /// atoms (`dual_formula`) — `Some` only when EVERY assert converted
    /// exactly (fail-closed). Consumed by the relational formula-implication
    /// check that authorizes the difference-network `unsat`.
    pub formula_dnf: Option<crate::vnnlib::dual_formula::DualFormulaDnf>,
}

impl OutputConstraint {
    pub fn is_relational(&self) -> bool {
        matches!(
            self,
            OutputConstraint::LessEq(_, _)
                | OutputConstraint::GreaterEq(_, _)
                | OutputConstraint::LessThan(_, _)
                | OutputConstraint::GreaterThan(_, _)
        )
    }

    /// Returns the maximum output index referenced by this constraint.
    pub fn max_output_index(&self) -> usize {
        match self {
            OutputConstraint::LessEq(i, j)
            | OutputConstraint::GreaterEq(i, j)
            | OutputConstraint::LessThan(i, j)
            | OutputConstraint::GreaterThan(i, j) => (*i).max(*j),
            OutputConstraint::LessEqConst(i, _)
            | OutputConstraint::GreaterEqConst(i, _)
            | OutputConstraint::LessThanConst(i, _)
            | OutputConstraint::GreaterThanConst(i, _) => *i,
        }
    }

    /// Every output index this constraint reads (one for the `*Const` forms,
    /// two for the relational forms).
    pub fn referenced_output_indices(&self, out: &mut Vec<usize>) {
        match self {
            OutputConstraint::LessEq(i, j)
            | OutputConstraint::GreaterEq(i, j)
            | OutputConstraint::LessThan(i, j)
            | OutputConstraint::GreaterThan(i, j) => {
                out.push(*i);
                out.push(*j);
            }
            OutputConstraint::LessEqConst(i, _)
            | OutputConstraint::GreaterEqConst(i, _)
            | OutputConstraint::LessThanConst(i, _)
            | OutputConstraint::GreaterThanConst(i, _) => out.push(*i),
        }
    }
}

/// A parsed VNN-LIB specification.
#[derive(Debug, Clone)]
pub struct VnnLibSpec {
    /// Number of input variables (X_0, X_1, ..., X_{n-1}).
    pub num_inputs: usize,
    /// Number of output variables (Y_0, Y_1, ..., Y_{m-1}).
    pub num_outputs: usize,
    /// Input bounds as (lower, upper) for each X_i.
    pub input_bounds: Vec<(f64, f64)>,
    /// Output constraints.
    pub output_constraints: Vec<OutputConstraint>,
    /// Output constraints grouped by disjunctive clauses.
    /// Each clause is a conjunction; the unsafe region is the OR of clauses.
    pub output_constraint_clauses: Vec<Vec<OutputConstraint>>,
    /// Whether output constraints form a disjunction (OR) at the top level.
    /// If true, unsafe region is (C1 OR C2 OR ...), so SAFE requires ALL violated.
    /// If false, unsafe region is (C1 AND C2 AND ...), so SAFE requires ANY violated.
    pub is_disjunction: bool,
    /// VNN-LIB version if declared via `(vnnlib-version X.Y)`.
    /// None if no version declaration was found (assumed VNN-LIB 1.0).
    pub version: Option<String>,
    /// Per-clause input bounds for mixed input+output disjunctive properties.
    /// Parallel to `output_constraint_clauses`. Each entry maps input variable
    /// index to (lower, upper) bounds specific to that clause. Empty map means
    /// use global `input_bounds`. Used by nn4sys lindex benchmarks.
    pub per_clause_input_bounds: Vec<std::collections::BTreeMap<usize, (f64, f64)>>,
    /// The TOP-LEVEL (non-clause) declared input bounds, captured BEFORE the
    /// per-clause union widening of `input_bounds` (see the parser's
    /// `apply_normalized_output_constraints`). A top-level assert constrains
    /// EVERY clause, so witness-membership gates must enforce these bounds in
    /// addition to any per-clause box — `input_bounds` alone cannot serve:
    /// it is widened to the clause union, discarding tighter declared values.
    /// Empty for programmatically built specs (no declared asserts).
    pub declared_input_bounds: Vec<(f64, f64)>,
    /// VNN-LIB 2.0 dual-network relation, when the property declares two networks.
    pub dual_network: Option<DualNetworkSpec>,
}

/// Convert an f64 rhs value UP for a sound, tight binary32 overapproximation.
///
/// This must classify finite overflow before hardware narrowing: positive
/// overflow widens to `+inf`, while negative overflow has the tighter valid
/// upper endpoint `f32::MIN`. A finite sentinel chosen without regard to sign
/// can shrink `A*y <= rhs` and manufacture an empty candidate-violation region.
/// See #2360, #2658.
fn rhs_with_overflow_guard(v: f64) -> f32 {
    ny_core::f64_to_f32_up(v)
}

impl VnnLibSpec {
    /// Create a new empty VNN-LIB specification.
    pub fn new() -> Self {
        Self {
            num_inputs: 0,
            num_outputs: 0,
            input_bounds: Vec::new(),
            output_constraints: Vec::new(),
            output_constraint_clauses: Vec::new(),
            is_disjunction: false,
            version: None,
            per_clause_input_bounds: Vec::new(),
            declared_input_bounds: Vec::new(),
            dual_network: None,
        }
    }

    /// Validate that all output constraint indices are within `[0, num_outputs)`.
    ///
    /// Returns an error if any `OutputConstraint` references an output index
    /// that is >= `num_outputs`. This catches malformed VNN-LIB specs at parse
    /// time, preventing downstream `InvalidSpec` errors (in `to_output_constraints`) or
    /// silently weakened objectives (in `build_relational_objective`).
    /// Every OUTPUT index this specification reads, sorted and deduplicated.
    ///
    /// The union spans the flat constraint list AND every disjunctive clause,
    /// because a single bound collection serves all of them: a row any clause
    /// reads must be tightened, whichever clause ends up deciding the verdict.
    ///
    /// This is what `#margin-subset-seed` needs to seed only the rows the
    /// verdict can read. On TinyYOLO (yolo_2023) the spec constrains 5 of
    /// 21,125 outputs, so the full `[21125 x 21125]` identity seed the OUTPUT
    /// node would otherwise take is 4,225x larger than required.
    pub fn referenced_output_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        for constraint in &self.output_constraints {
            constraint.referenced_output_indices(&mut indices);
        }
        for clause in &self.output_constraint_clauses {
            for constraint in clause {
                constraint.referenced_output_indices(&mut indices);
            }
        }
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    pub fn validate_output_indices(&self) -> ny_core::Result<()> {
        let num = self.num_outputs;
        let check = |constraint: &OutputConstraint| -> ny_core::Result<()> {
            let max_idx = constraint.max_output_index();
            if max_idx >= num {
                let range_desc = if num == 0 {
                    "no outputs declared".to_string()
                } else {
                    format!("{} outputs declared (Y_0..Y_{})", num, num - 1)
                };
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "Output constraint references Y_{} but {}",
                    max_idx, range_desc,
                )));
            }
            Ok(())
        };

        for c in &self.output_constraints {
            check(c)?;
        }
        for clause in &self.output_constraint_clauses {
            for c in clause {
                check(c)?;
            }
        }
        Ok(())
    }

    /// Validate that every input variable's bounds are well-formed.
    ///
    /// Rejects NaN bounds (which a literal like `NaN` in a VNN-LIB `assert`
    /// produces) and inverted intervals where `lower > upper`. Unbounded inputs
    /// (±∞ on an unconstrained variable) are allowed. Catching this at the
    /// VNN-LIB boundary — where the original `X_i` variable names are still
    /// known — yields a diagnostic that names the offending variable, instead of
    /// the downstream numeric `input_bounds[i]` index error from the core spec
    /// validator (#2800).
    pub fn validate_input_bounds(&self) -> ny_core::Result<()> {
        for (i, (lower, upper)) in self.input_bounds.iter().enumerate() {
            if lower.is_nan() || upper.is_nan() {
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "Input variable X_{i} has an invalid (NaN) bound: [{lower}, {upper}]"
                )));
            }
            if lower > upper {
                return Err(ny_core::NyError::InvalidSpec(format!(
                    "Input variable X_{i} has an invalid bound: lower {lower} > upper {upper}"
                )));
            }
        }
        Ok(())
    }

    /// Check if the specification has valid input bounds.
    pub fn has_valid_bounds(&self) -> bool {
        self.input_bounds
            .iter()
            .all(|(lower, upper)| lower <= upper)
    }

    /// Split input bounds into separate lower/upper vectors.
    ///
    /// Projects the stored `(lower, upper)` pairs into two parallel vectors.
    pub fn split_input_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let lower: Vec<f64> = self.input_bounds.iter().map(|(l, _)| *l).collect();
        let upper: Vec<f64> = self.input_bounds.iter().map(|(_, u)| *u).collect();
        (lower, upper)
    }

    /// Deprecated compatibility alias for [`split_input_bounds`](Self::split_input_bounds).
    #[deprecated(note = "use split_input_bounds")]
    pub fn get_input_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        self.split_input_bounds()
    }

    /// Split input bounds into f32 vectors with directed rounding for soundness.
    ///
    /// Lower bounds are rounded toward -inf (`next_down_f32`) and upper bounds
    /// toward +inf (`next_up_f32`) so the f32 region is a superset of the f64
    /// region specified in the VNN-LIB file. Plain `as f32` round-to-nearest
    /// could shrink the verified region and miss counterexamples (#2658).
    pub fn split_input_bounds_f32(&self) -> (Vec<f32>, Vec<f32>) {
        use ny_tensor::{next_down_f32, next_up_f32};
        let lower: Vec<f32> = self
            .input_bounds
            .iter()
            .map(|(l, _)| {
                // #2360: Guard f64→f32 overflow. Values beyond f32 range silently
                // become ±inf; use -f32::MAX as the most conservative finite lower.
                let v = *l as f32;
                if v.is_finite() {
                    next_down_f32(v)
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let upper: Vec<f32> = self
            .input_bounds
            .iter()
            .map(|(_, u)| {
                let v = *u as f32;
                if v.is_finite() {
                    next_up_f32(v)
                } else {
                    f32::INFINITY
                }
            })
            .collect();
        (lower, upper)
    }

    /// Deprecated compatibility alias for [`split_input_bounds_f32`](Self::split_input_bounds_f32).
    #[deprecated(note = "use split_input_bounds_f32")]
    pub fn get_input_bounds_f32(&self) -> (Vec<f32>, Vec<f32>) {
        self.split_input_bounds_f32()
    }

    /// Returns true if output constraints are compatible with peeling off softmax/sigmoid.
    ///
    /// This requires all output constraints to be purely relational (Y_i ? Y_j).
    /// Constraints involving constants are not supported because the monotonic
    /// logit transform is non-linear and would require rewriting thresholds.
    pub fn supports_logits_peel(&self) -> bool {
        if self.output_constraint_clauses.is_empty() {
            self.output_constraints
                .iter()
                .all(OutputConstraint::is_relational)
        } else {
            self.output_constraint_clauses
                .iter()
                .flatten()
                .all(OutputConstraint::is_relational)
        }
    }

    /// Returns true if the output satisfies ALL constraints (i.e., is in the unsafe region).
    pub fn is_unsafe(&self, outputs: &[f64]) -> bool {
        if outputs.len() < self.num_outputs {
            return false;
        }

        let output = |idx: usize| outputs.get(idx).copied();
        let satisfies = |constraint: &OutputConstraint| match constraint {
            OutputConstraint::LessEq(i, j) => match (output(*i), output(*j)) {
                (Some(a), Some(b)) => a <= b,
                _ => false,
            },
            OutputConstraint::GreaterEq(i, j) => match (output(*i), output(*j)) {
                (Some(a), Some(b)) => a >= b,
                _ => false,
            },
            OutputConstraint::LessThan(i, j) => match (output(*i), output(*j)) {
                (Some(a), Some(b)) => a < b,
                _ => false,
            },
            OutputConstraint::GreaterThan(i, j) => match (output(*i), output(*j)) {
                (Some(a), Some(b)) => a > b,
                _ => false,
            },
            OutputConstraint::LessEqConst(i, c) => match output(*i) {
                Some(a) => a <= *c,
                None => false,
            },
            OutputConstraint::GreaterEqConst(i, c) => match output(*i) {
                Some(a) => a >= *c,
                None => false,
            },
            OutputConstraint::LessThanConst(i, c) => match output(*i) {
                Some(a) => a < *c,
                None => false,
            },
            OutputConstraint::GreaterThanConst(i, c) => match output(*i) {
                Some(a) => a > *c,
                None => false,
            },
        };

        if self.output_constraint_clauses.is_empty() {
            if self.output_constraints.is_empty() {
                return false;
            }
            return self.output_constraints.iter().all(satisfies);
        }

        if self.is_disjunction {
            self.output_constraint_clauses
                .iter()
                .any(|clause| clause.iter().all(&satisfies))
        } else {
            self.output_constraint_clauses
                .iter()
                .all(|clause| clause.iter().all(&satisfies))
        }
    }

    /// Describe the property in human-readable form.
    pub fn describe(&self) -> String {
        let mut desc = format!(
            "VNN-LIB Property: {} inputs, {} outputs\n",
            self.num_inputs, self.num_outputs
        );

        desc.push_str("Input bounds:\n");
        for (i, (lower, upper)) in self.input_bounds.iter().enumerate() {
            desc.push_str(&format!("  X_{}: [{:.6}, {:.6}]\n", i, lower, upper));
        }

        if self.is_disjunction {
            desc.push_str("Output constraints (unsafe if ANY clause satisfied):\n");
        } else {
            desc.push_str("Output constraints (unsafe if ALL satisfied):\n");
        }

        let clauses: Vec<&[OutputConstraint]> = if self.output_constraint_clauses.is_empty() {
            if self.output_constraints.is_empty() {
                Vec::new()
            } else {
                vec![self.output_constraints.as_slice()]
            }
        } else {
            self.output_constraint_clauses
                .iter()
                .map(|c| c.as_slice())
                .collect()
        };

        for (idx, clause) in clauses.iter().enumerate() {
            if self.is_disjunction {
                desc.push_str(&format!("  Clause {}:\n", idx + 1));
            }
            for c in *clause {
                match c {
                    OutputConstraint::LessEq(i, j) => {
                        desc.push_str(&format!("    Y_{} <= Y_{}\n", i, j))
                    }
                    OutputConstraint::GreaterEq(i, j) => {
                        desc.push_str(&format!("    Y_{} >= Y_{}\n", i, j))
                    }
                    OutputConstraint::LessThan(i, j) => {
                        desc.push_str(&format!("    Y_{} < Y_{}\n", i, j))
                    }
                    OutputConstraint::GreaterThan(i, j) => {
                        desc.push_str(&format!("    Y_{} > Y_{}\n", i, j))
                    }
                    OutputConstraint::LessEqConst(i, c) => {
                        desc.push_str(&format!("    Y_{} <= {:.6}\n", i, c))
                    }
                    OutputConstraint::GreaterEqConst(i, c) => {
                        desc.push_str(&format!("    Y_{} >= {:.6}\n", i, c))
                    }
                    OutputConstraint::LessThanConst(i, c) => {
                        desc.push_str(&format!("    Y_{} < {:.6}\n", i, c))
                    }
                    OutputConstraint::GreaterThanConst(i, c) => {
                        desc.push_str(&format!("    Y_{} > {:.6}\n", i, c))
                    }
                }
            }
        }

        desc
    }

    /// Convert output constraints to INVPROP matrix form.
    ///
    /// Transforms the VNN-LIB output constraints into the matrix representation
    /// `A * y <= rhs` used by INVPROP for output constraint backward propagation.
    ///
    /// # Constraint Mapping
    ///
    /// Each `OutputConstraint` variant maps to `A * y <= rhs` as follows:
    /// - `LessEq(i, j)`: `Y_i - Y_j <= 0` → row `[..., +1@i, ..., -1@j, ...]`, rhs=0
    /// - `GreaterEq(i, j)`: `Y_j - Y_i <= 0` → row `[..., -1@i, ..., +1@j, ...]`, rhs=0
    /// - `LessThan(i, j)`: treated as `LessEq` for soundness (strict → non-strict)
    /// - `GreaterThan(i, j)`: treated as `GreaterEq` for soundness
    /// - `LessEqConst(i, c)`: `Y_i <= c` → row `[..., +1@i, ...]`, rhs=c
    /// - `GreaterEqConst(i, c)`: `-Y_i <= -c` → row `[..., -1@i, ...]`, rhs=-c
    /// - `LessThanConst(i, c)`: treated as `LessEqConst` for soundness
    /// - `GreaterThanConst(i, c)`: treated as `GreaterEqConst` for soundness
    ///
    /// # Soundness Note
    ///
    /// Strict inequalities (`<`, `>`) are relaxed to non-strict (`<=`, `>=`) for
    /// soundness in verification. This means the verifier may report "safe" for
    /// boundary cases where the strict inequality fails but non-strict passes.
    /// This is conservative (never misses a violation in the original spec).
    ///
    /// # Returns
    ///
    /// `OutputConstraints` in matrix form, suitable for INVPROP.
    /// Returns constraints with `is_conjunction` matching `!self.is_disjunction`.
    /// For disjunctions with multi-constraint clauses, clause groupings are
    /// preserved in `OutputConstraints::clause_indices`.
    ///
    /// # Errors
    ///
    /// Returns `NyError::InvalidSpec` if `num_outputs == 0`.
    pub fn to_output_constraints(&self) -> ny_core::Result<ny_propagate::OutputConstraints> {
        use ndarray::{Array1, Array2};

        if self.num_outputs == 0 {
            return Err(ny_core::NyError::InvalidSpec(
                "Cannot convert to OutputConstraints: num_outputs is 0".to_string(),
            ));
        }
        // `VnnLibSpec` is public and can be constructed programmatically without
        // passing through the parser's validation. Preserve this Result API at
        // the matrix-construction boundary instead of indexing a malformed Y_i
        // and panicking.
        self.validate_output_indices()?;
        let clauses: Vec<&[OutputConstraint]> = if self.output_constraint_clauses.is_empty() {
            if self.output_constraints.is_empty() {
                Vec::new()
            } else {
                vec![self.output_constraints.as_slice()]
            }
        } else {
            self.output_constraint_clauses
                .iter()
                .map(|c| c.as_slice())
                .collect()
        };

        let mut flat_constraints: Vec<&OutputConstraint> = Vec::new();
        let mut clause_indices: Vec<Vec<usize>> = Vec::new();

        for clause in &clauses {
            let mut indices = Vec::with_capacity(clause.len());
            for constraint in *clause {
                indices.push(flat_constraints.len());
                flat_constraints.push(constraint);
            }
            clause_indices.push(indices);
        }

        let num_constraints = flat_constraints.len();
        let output_dim = self.num_outputs;

        let mut a_matrix = Array2::<f32>::zeros((num_constraints, output_dim));
        let mut rhs = Array1::<f32>::zeros(num_constraints);

        for (row, constraint) in flat_constraints.iter().enumerate() {
            match constraint {
                OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                    // Y_i - Y_j <= 0
                    a_matrix[[row, *i]] = 1.0;
                    a_matrix[[row, *j]] = -1.0;
                    rhs[row] = 0.0;
                }
                OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                    // Y_j - Y_i <= 0 (equivalently: -Y_i + Y_j <= 0)
                    a_matrix[[row, *i]] = -1.0;
                    a_matrix[[row, *j]] = 1.0;
                    rhs[row] = 0.0;
                }
                OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
                    // Y_i <= c — round rhs UP to widen (sound). See #2658.
                    a_matrix[[row, *i]] = 1.0;
                    rhs[row] = rhs_with_overflow_guard(*c);
                }
                OutputConstraint::GreaterEqConst(i, c)
                | OutputConstraint::GreaterThanConst(i, c) => {
                    // -Y_i <= -c — round rhs UP (i.e., -c toward +inf). See #2658.
                    a_matrix[[row, *i]] = -1.0;
                    rhs[row] = rhs_with_overflow_guard(-*c);
                }
            }
        }

        // VNN-LIB is_disjunction=true means OR (any one satisfied = unsafe)
        // INVPROP is_conjunction=true means AND (all must be satisfied)
        // They are inverses in semantic meaning for the unsafe region
        let mut constraints =
            ny_propagate::OutputConstraints::new(a_matrix, rhs, !self.is_disjunction)?;
        if self.is_disjunction && !clause_indices.is_empty() {
            constraints.clause_indices = Some(clause_indices);
        }
        Ok(constraints)
    }

    /// Shrink input bounds inward by `eps` per dimension (lower += eps, upper -= eps).
    /// Reference: alpha-beta-CROWN `shrink_vnnlib` (`specifications.py:535-540`).
    pub fn shrink_input_bounds(&mut self, eps: f64) {
        assert!(
            eps.is_finite() && eps > 0.0,
            "shrink_eps must be positive and finite"
        );
        for (lower, upper) in &mut self.input_bounds {
            *lower += eps;
            *upper -= eps;
        }
        for clause_map in &mut self.per_clause_input_bounds {
            for (lower, upper) in clause_map.values_mut() {
                *lower += eps;
                *upper -= eps;
            }
        }
    }

    /// Returns true if the property is a multi-clause disjunction (2+ clauses)
    /// requiring per-clause verification dispatch.
    pub fn has_multi_constraint_disjunction(&self) -> bool {
        self.is_disjunction && self.output_constraint_clauses.len() >= 2
    }

    /// Returns true if the property is a disjunction of clauses that carry
    /// their OWN per-clause input boxes (the nn4sys mscn/lindex band shape:
    /// `(or (and <input box> <output constraint>) ...)`), with every clause
    /// retaining at least one output constraint after input-atom stripping.
    ///
    /// Unlike `has_multi_constraint_disjunction`, this is true also for a
    /// SINGLE such clause (mscn `_dual` cardinality_1_1): one-clause
    /// disjunctive semantics — UNSAT iff that clause is impossible over its
    /// own box — coincide exactly with the conjunctive reading of the same
    /// clause, and the per-clause-box refinement screen is the lane equipped
    /// to decide it (the global-box path sees the identical input hull but
    /// lacks the box-refinement + f64 leaf escalation engine).
    pub fn has_boxed_clause_disjunction(&self) -> bool {
        !self.output_constraint_clauses.is_empty()
            && self.output_constraint_clauses.iter().all(|c| !c.is_empty())
            && self.per_clause_input_bounds.iter().any(|b| !b.is_empty())
    }
}

impl Default for VnnLibSpec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod referenced_output_indices_tests {
    use super::{OutputConstraint, VnnLibSpec};

    /// The union must span BOTH the flat constraint list and every disjunctive
    /// clause: one bound collection serves all clauses, so a row any clause
    /// reads must be tightened. Missing a clause would leave that row on its
    /// looser IBP bound — still sound, but it silently gives up the tightening
    /// this seed exists to provide.
    #[test]
    fn union_spans_clauses_and_relational_forms_and_dedupes() {
        let mut spec = VnnLibSpec::new();
        spec.num_outputs = 21_125;
        spec.output_constraints = vec![
            OutputConstraint::LessEqConst(776, -1.0),
            OutputConstraint::LessEqConst(100, -1.0),
            // Duplicate: must collapse.
            OutputConstraint::LessEqConst(100, -2.0),
        ];
        spec.output_constraint_clauses = vec![
            vec![OutputConstraint::GreaterEq(269, 438)],
            vec![OutputConstraint::LessThan(607, 100)],
        ];

        // Sorted, deduplicated, and BOTH operands of each relational form.
        assert_eq!(
            spec.referenced_output_indices(),
            vec![100, 269, 438, 607, 776]
        );
    }

    /// A spec with no output constraints publishes nothing, which disengages
    /// subset seeding and keeps the historical full-width path byte-identical.
    #[test]
    fn empty_spec_references_nothing() {
        let spec = VnnLibSpec::new();
        assert!(spec.referenced_output_indices().is_empty());
    }

    /// The TinyYOLO (yolo_2023) shape this was built for: 5 of 21,125 outputs.
    /// The full identity seed the OUTPUT node would otherwise take is
    /// 8 * 21125^2 = 3.57 GB, which the Conv2d scratch cap refuses.
    #[test]
    fn tinyyolo_spec_reads_five_of_21125_outputs() {
        let mut spec = VnnLibSpec::new();
        spec.num_outputs = 21_125;
        spec.output_constraints = [100, 269, 438, 607, 776]
            .into_iter()
            .map(|i| OutputConstraint::LessEqConst(i, -1.0))
            .collect();
        let referenced = spec.referenced_output_indices();
        assert_eq!(referenced.len(), 5);
        assert_eq!(referenced.last(), Some(&776));
        assert!(referenced.len() < spec.num_outputs);
    }
}
