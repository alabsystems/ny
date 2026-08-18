// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// MIP encoder for FC+ReLU networks targeting the solver-neutral MILP IR.
//
// Structurally parallel to 's NetworkEncoder (encoder/layers.rs),
// but targets the `MilpProblem` IR (lowered per-backend at solve time)
// instead of SMT-LIB text.
// Big-M formulation matches 's encode_relu_bigm (layers.rs:298).

use crate::error::MipError;
use crate::ir::{Col, MilpProblem, Row};
use num_rational::BigRational;
use ny_core::Bound;

type Result<T> = std::result::Result<T, MipError>;

/// Encodes a sequential FC+ReLU network as a MIP/LP problem.
///
/// Variables are tracked by IR [`Col`] indices. The encoder maintains
/// a "frontier" of current variables that get wired into the next layer.
///
/// `Clone` lets disjunctive callers encode the (identical) network once and
/// stamp per-clause output constraints onto cheap clones: the clone copies the
/// built IR (a few MB of sparse rows), while a re-encode re-scans the dense
/// weight matrices (252M f64 for the malbeware 16-25 conv unfold) per clause.
#[derive(Clone)]
pub struct MipEncoder {
    problem: MilpProblem,
    /// Variable indices for network input neurons.
    input_vars: Vec<Col>,
    /// Variable indices for network output neurons (set by `finalize`).
    output_vars: Vec<Col>,
    /// Variable indices for ReLU binary indicators (for diagnostics).
    binary_vars: Vec<Col>,
    /// Pre-activation width `u - l` per unstable neuron, aligned with
    /// `binary_vars`. Phase-split racing branches on the widest binaries
    /// (widest pre-activation range = least constrained). designs/scip.md.
    binary_widths: Vec<f64>,
    /// Current frontier: output of the last-encoded layer.
    current_vars: Vec<Col>,
}

impl MipEncoder {
    /// Create an encoder with input variables bounded by `input_bounds`.
    ///
    /// Each input gets a continuous variable with objective coefficient 0.
    pub fn new(input_bounds: &[Bound]) -> Result<Self> {
        let mut problem = MilpProblem::new();
        let mut input_vars = Vec::with_capacity(input_bounds.len());

        for (i, b) in input_bounds.iter().enumerate() {
            if b.lower().is_nan() || b.upper().is_nan() {
                return Err(MipError::InvalidBounds(format!(
                    "NaN bound for input variable {i}"
                )));
            }
            let col = problem.add_col(0.0, b.lower() as f64, b.upper() as f64);
            input_vars.push(col);
        }

        Ok(Self {
            current_vars: input_vars.clone(),
            input_vars,
            output_vars: Vec::new(),
            binary_vars: Vec::new(),
            binary_widths: Vec::new(),
            problem,
        })
    }

    /// Create an encoder with binary64 input bounds preserved exactly.
    ///
    /// This is the authority-preserving constructor for frontends whose source
    /// property stores finite binary64 endpoints. Routing those endpoints
    /// through [`Bound`] would first widen/narrow them to binary32 and can erase
    /// a small exact-LP separation. The ordinary [`Self::new`] behavior remains
    /// unchanged for existing callers.
    pub fn new_with_f64_bounds(input_bounds: &[(f64, f64)]) -> Result<Self> {
        let mut problem = MilpProblem::new();
        let mut input_vars = Vec::with_capacity(input_bounds.len());

        for (i, &(lower, upper)) in input_bounds.iter().enumerate() {
            if !lower.is_finite() || !upper.is_finite() {
                return Err(MipError::InvalidBounds(format!(
                    "non-finite bound for input variable {i}: [{lower}, {upper}]"
                )));
            }
            if lower > upper {
                return Err(MipError::InvalidBounds(format!(
                    "inverted bound for input variable {i}: {lower} > {upper}"
                )));
            }
            let col = problem.add_col(0.0, lower, upper);
            input_vars.push(col);
        }

        Ok(Self {
            current_vars: input_vars.clone(),
            input_vars,
            output_vars: Vec::new(),
            binary_vars: Vec::new(),
            binary_widths: Vec::new(),
            problem,
        })
    }

    /// Encode a fully connected (linear) layer: y = Wx + b.
    ///
    /// `weights` is row-major [out_features × in_features].
    /// `bias` has length out_features.
    ///
    /// Reference:  encoder/layers.rs:178 (encode_linear)
    pub fn encode_linear(
        &mut self,
        weights: &[f64],
        bias: &[f64],
        out_features: usize,
    ) -> Result<()> {
        let in_features = self.current_vars.len();
        if weights.len() != out_features * in_features {
            return Err(MipError::Encoding(format!(
                "weight matrix size mismatch: expected {}x{} = {}, got {}",
                out_features,
                in_features,
                out_features * in_features,
                weights.len()
            )));
        }
        if bias.len() != out_features {
            return Err(MipError::Encoding(format!(
                "bias size mismatch: expected {}, got {}",
                out_features,
                bias.len()
            )));
        }

        let mut new_vars = Vec::with_capacity(out_features);
        for i in 0..out_features {
            // y_i is a free variable (unbounded continuous).
            let y_var = self.problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);

            // Constraint: y_i = sum_j(w_ij * x_j) + b_i
            // Rearranged: -y_i + sum_j(w_ij * x_j) = -b_i
            // IR form: sum of (col, coeff) pairs, row bounds [rhs, rhs]
            let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(in_features + 1);
            for (j, &x_var) in self.current_vars.iter().enumerate() {
                let w = weights[i * in_features + j];
                if w != 0.0 {
                    coeffs.push((x_var, w));
                }
            }
            coeffs.push((y_var, -1.0));

            // Equality: sum(w_ij * x_j) - y_i = -b_i
            let neg_b = -bias[i];
            self.problem.add_row(neg_b, neg_b, coeffs);

            new_vars.push(y_var);
        }

        self.current_vars = new_vars;
        Ok(())
    }

    /// Encode ReLU activation using Big-M formulation.
    ///
    /// For each neuron with pre-activation bounds [l, u]:
    /// - l >= 0: y = x (always active, no binary needed)
    /// - u <= 0: y = 0 (always inactive, no binary needed)
    /// - Otherwise (unstable): Big-M encoding with binary indicator z
    ///   - y >= 0
    ///   - y >= x
    ///   - y <= x - l*(1-z) = x + l*z - l
    ///   - y <= u*z
    ///   - z in {0, 1}
    ///
    /// The design uses tight Big-M values (l, u) from intermediate bounds
    /// rather than a global Big-M constant. This is tighter than 's
    /// fixed big_m=1e6 approach and matches the standard MILP formulation.
    ///
    /// Reference:  encoder/layers.rs:298 (encode_relu_bigm)
    /// Reference: design doc designs/2026-03-04-highs-mip-solver-integration.md
    pub fn encode_relu(&mut self, pre_activation_bounds: &[Bound]) -> Result<()> {
        let n = self.current_vars.len();
        if pre_activation_bounds.len() != n {
            return Err(MipError::Encoding(format!(
                "ReLU bounds dimension mismatch: {} vars, {} bounds",
                n,
                pre_activation_bounds.len()
            )));
        }

        let mut new_vars = Vec::with_capacity(n);
        for (i, bound) in pre_activation_bounds.iter().enumerate() {
            let x_var = self.current_vars[i];
            let lb = bound.lower() as f64;
            let ub = bound.upper() as f64;

            if lb >= 0.0 {
                // Always active: y = x (no new variable needed)
                new_vars.push(x_var);
            } else if ub <= 0.0 {
                // Always inactive: y = 0 (fixed variable)
                let y_var = self.problem.add_col(0.0, 0.0, 0.0);
                new_vars.push(y_var);
            } else {
                // Unstable neuron: Big-M encoding with tight bounds
                let y_var = self.problem.add_col(0.0, 0.0, ub);
                let z_var = self.problem.add_integer_col(0.0, 0.0, 1.0);
                self.binary_vars.push(z_var);
                self.binary_widths.push(ub - lb);

                // y >= x  =>  y - x >= 0
                self.problem
                    .add_row(0.0, f64::INFINITY, [(y_var, 1.0), (x_var, -1.0)]);

                // y <= x - l*(1-z) = x + l*z - l
                // Rearranged: y - x - l*z <= -l
                self.problem.add_row(
                    f64::NEG_INFINITY,
                    -lb,
                    [(y_var, 1.0), (x_var, -1.0), (z_var, -lb)],
                );

                // y <= u*z  =>  y - u*z <= 0
                self.problem
                    .add_row(f64::NEG_INFINITY, 0.0, [(y_var, 1.0), (z_var, -ub)]);

                new_vars.push(y_var);
            }
        }

        self.current_vars = new_vars;
        Ok(())
    }

    /// Encode the exact continuous Planet outer relaxation of a ReLU layer.
    ///
    /// Stable coordinates retain the exact encodings used by [`Self::encode_relu`].
    /// For an unstable coordinate with finite `l < 0 < u`, this adds no binary
    /// variable and encodes the convex hull
    ///
    /// ```text
    /// y >= 0
    /// y >= x
    /// (u - l)y - ux <= -ul.
    /// ```
    ///
    /// The first inequality is the lower bound of the new `y` column. The two
    /// arithmetic expressions in the upper-hull row are formed as exact
    /// rationals first. If the corresponding binary64 subtraction or product
    /// changes that exact value, the whole layer is rejected before the
    /// encoder is mutated. This is intentionally stricter than an outward
    /// rounded relaxation: an exact Farkas replay must describe precisely the
    /// hull that this API claims to have encoded.
    pub fn encode_relu_continuous_outer(
        &mut self,
        pre_activation_bounds: &[(f64, f64)],
    ) -> Result<()> {
        let n = self.current_vars.len();
        if pre_activation_bounds.len() != n {
            return Err(MipError::Encoding(format!(
                "continuous ReLU bounds dimension mismatch: {} vars, {} bounds",
                n,
                pre_activation_bounds.len()
            )));
        }

        // Validate every exact coefficient before adding any column or row so
        // an inexact late coordinate cannot leave a partially encoded layer.
        let plans = pre_activation_bounds
            .iter()
            .enumerate()
            .map(|(index, &(lower, upper))| continuous_relu_plan(index, lower, upper))
            .collect::<Result<Vec<_>>>()?;

        let mut new_vars = Vec::with_capacity(n);
        for (x_var, plan) in self.current_vars.iter().copied().zip(plans) {
            match plan {
                ContinuousReluPlan::Active => new_vars.push(x_var),
                ContinuousReluPlan::Inactive => {
                    let y_var = self.problem.add_col(0.0, 0.0, 0.0);
                    new_vars.push(y_var);
                }
                ContinuousReluPlan::Unstable {
                    upper,
                    width,
                    upper_rhs,
                } => {
                    // y >= 0 is represented by the exact column lower bound.
                    let y_var = self.problem.add_col(0.0, 0.0, upper);

                    // y >= x.
                    self.problem
                        .add_row(0.0, f64::INFINITY, [(y_var, 1.0), (x_var, -1.0)]);

                    // (u-l)y - ux <= -ul.
                    self.problem.add_row(
                        f64::NEG_INFINITY,
                        upper_rhs,
                        [(y_var, width), (x_var, -upper)],
                    );
                    new_vars.push(y_var);
                }
            }
        }

        self.current_vars = new_vars;
        Ok(())
    }

    /// Mark the current frontier as output variables.
    pub fn finalize(&mut self) {
        self.output_vars = self.current_vars.clone();
    }

    /// Add constraint: `output[i] <= output[j]`.
    ///
    /// Encodes as: Y_i - Y_j <= 0.
    /// Used for VNNLIB LessEq(i, j) constraints.
    pub fn constrain_output_leq(&mut self, i: usize, j: usize) -> Result<()> {
        self.constrain_output_leq_row(i, j)?;
        Ok(())
    }

    /// Add `output[i] <= output[j]` and return the exact emitted row.
    ///
    /// The row identity lets an explicitly gated caller optimize this same
    /// one-sided unsafe-region constraint for a robust SAT candidate without
    /// guessing from row shape.  It grants no verdict authority.
    pub fn constrain_output_leq_row(&mut self, i: usize, j: usize) -> Result<Row> {
        let (yi, yj) = self.output_pair(i, j)?;
        Ok(self
            .problem
            .add_row(f64::NEG_INFINITY, 0.0, [(yi, 1.0), (yj, -1.0)]))
    }

    /// Add constraint: `output[i] >= output[j]`.
    ///
    /// Encodes as: Y_i - Y_j >= 0.
    /// Used for VNNLIB GreaterEq(i, j) constraints.
    pub fn constrain_output_geq(&mut self, i: usize, j: usize) -> Result<()> {
        self.constrain_output_geq_row(i, j)?;
        Ok(())
    }

    /// Add `output[i] >= output[j]` and return the exact emitted row.
    ///
    /// See [`Self::constrain_output_leq_row`] for the witness-only use of this
    /// identity.
    pub fn constrain_output_geq_row(&mut self, i: usize, j: usize) -> Result<Row> {
        let (yi, yj) = self.output_pair(i, j)?;
        Ok(self
            .problem
            .add_row(0.0, f64::INFINITY, [(yi, 1.0), (yj, -1.0)]))
    }

    /// Add constraint: `output[i] <= constant`.
    ///
    /// Used for VNNLIB LessEqConst(i, c) constraints.
    pub fn constrain_output_leq_const(&mut self, i: usize, val: f64) -> Result<()> {
        self.constrain_output_leq_const_row(i, val)?;
        Ok(())
    }

    /// Add `output[i] <= val` and return the exact emitted row.
    pub fn constrain_output_leq_const_row(&mut self, i: usize, val: f64) -> Result<Row> {
        let yi = self.output_var(i)?;
        Ok(self.problem.add_row(f64::NEG_INFINITY, val, [(yi, 1.0)]))
    }

    /// Add constraint: `output[i] >= constant`.
    ///
    /// Used for VNNLIB GreaterEqConst(i, c) constraints.
    pub fn constrain_output_geq_const(&mut self, i: usize, val: f64) -> Result<()> {
        self.constrain_output_geq_const_row(i, val)?;
        Ok(())
    }

    /// Add `output[i] >= val` and return the exact emitted row.
    pub fn constrain_output_geq_const_row(&mut self, i: usize, val: f64) -> Result<Row> {
        let yi = self.output_var(i)?;
        Ok(self.problem.add_row(val, f64::INFINITY, [(yi, 1.0)]))
    }

    /// Look up a single output variable by index.
    fn output_var(&self, i: usize) -> Result<Col> {
        self.output_vars.get(i).copied().ok_or_else(|| {
            MipError::Encoding(format!(
                "output index {} out of range ({})",
                i,
                self.output_vars.len()
            ))
        })
    }

    /// Look up a pair of output variables by index.
    fn output_pair(&self, i: usize, j: usize) -> Result<(Col, Col)> {
        Ok((self.output_var(i)?, self.output_var(j)?))
    }

    /// Total number of columns (variables) in the encoded problem.
    pub fn num_cols(&self) -> usize {
        self.problem.num_cols()
    }

    /// Consume the encoder and return the built MILP IR plus metadata.
    pub fn into_parts(self) -> MipParts {
        let num_cols = self.problem.num_cols();
        MipParts {
            problem: self.problem,
            input_vars: self.input_vars,
            output_vars: self.output_vars,
            binary_vars: self.binary_vars,
            binary_widths: self.binary_widths,
            num_cols,
        }
    }

    /// Get the number of binary (ReLU indicator) variables.
    pub fn num_binary_vars(&self) -> usize {
        self.binary_vars.len()
    }

    /// Get the output variable indices.
    pub fn output_vars(&self) -> &[Col] {
        &self.output_vars
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ContinuousReluPlan {
    Active,
    Inactive,
    Unstable {
        upper: f64,
        width: f64,
        upper_rhs: f64,
    },
}

fn continuous_relu_plan(index: usize, lower: f64, upper: f64) -> Result<ContinuousReluPlan> {
    let exact_lower = BigRational::from_float(lower).ok_or_else(|| {
        MipError::InvalidBounds(format!(
            "non-finite continuous ReLU lower bound at coordinate {index}: {lower}"
        ))
    })?;
    let exact_upper = BigRational::from_float(upper).ok_or_else(|| {
        MipError::InvalidBounds(format!(
            "non-finite continuous ReLU upper bound at coordinate {index}: {upper}"
        ))
    })?;
    if lower > upper {
        return Err(MipError::InvalidBounds(format!(
            "inverted continuous ReLU bound at coordinate {index}: {lower} > {upper}"
        )));
    }
    if lower >= 0.0 {
        return Ok(ContinuousReluPlan::Active);
    }
    if upper <= 0.0 {
        return Ok(ContinuousReluPlan::Inactive);
    }

    let exact_width = &exact_upper - &exact_lower;
    let width = upper - lower;
    require_exact_binary64_result(width, &exact_width, index, "unstable ReLU width u-l")?;

    let exact_upper_rhs = -(&exact_upper * &exact_lower);
    let upper_rhs = -(upper * lower);
    require_exact_binary64_result(
        upper_rhs,
        &exact_upper_rhs,
        index,
        "unstable ReLU product -u*l",
    )?;

    Ok(ContinuousReluPlan::Unstable {
        upper,
        width,
        upper_rhs,
    })
}

fn require_exact_binary64_result(
    encoded: f64,
    exact: &BigRational,
    index: usize,
    expression: &str,
) -> Result<()> {
    if !encoded.is_finite()
        || BigRational::from_float(encoded).is_none_or(|round_trip| round_trip != *exact)
    {
        return Err(MipError::Encoding(format!(
            "{expression} at coordinate {index} is not exactly representable as finite f64"
        )));
    }
    Ok(())
}

/// Extracted parts of an encoded MIP problem.
pub struct MipParts {
    /// The solver-neutral MILP formulation (lowered per-backend at solve time).
    pub problem: MilpProblem,
    /// Input variable column indices.
    pub input_vars: Vec<Col>,
    /// Output variable column indices.
    pub output_vars: Vec<Col>,
    /// Binary ReLU indicator column indices.
    pub binary_vars: Vec<Col>,
    /// Pre-activation width `u - l` per unstable neuron, aligned with
    /// `binary_vars`. Drives phase-split branching selection (designs/scip.md).
    pub binary_widths: Vec<f64>,
    /// Total number of columns in the problem (for warm-start vector sizing).
    pub num_cols: usize,
}

/// Encode a full feedforward FC+ReLU network into a MIP problem.
///
/// This is the top-level convenience function, parallel to 's
/// `NetworkEncoder::encode_feedforward` (feedforward.rs:22).
///
/// # Arguments
/// * `weights` - Weight matrices per layer, row-major [out × in]
/// * `biases` - Bias vectors per layer
/// * `layer_dims` - [input_dim, hidden1, ..., output_dim]
/// * `input_bounds` - Bounds on network inputs
/// * `intermediate_bounds` - Pre-activation bounds per hidden layer (for Big-M)
pub fn encode_feedforward(
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    layer_dims: &[usize],
    input_bounds: &[Bound],
    intermediate_bounds: &[Vec<Bound>],
) -> Result<MipEncoder> {
    let num_layers = weights.len();
    if num_layers == 0 {
        return Err(MipError::Encoding("empty network".into()));
    }
    if biases.len() != num_layers {
        return Err(MipError::Encoding(format!(
            "bias layer count {} doesn't match weight layer count {}",
            biases.len(),
            num_layers
        )));
    }
    if layer_dims.len() != num_layers + 1 {
        return Err(MipError::Encoding(format!(
            "layer_dims length {} doesn't match {} layers + 1",
            layer_dims.len(),
            num_layers
        )));
    }
    if input_bounds.len() != layer_dims[0] {
        return Err(MipError::Encoding(format!(
            "input_bounds length {} doesn't match input dim {}",
            input_bounds.len(),
            layer_dims[0]
        )));
    }
    let expected_intermediate_bounds = num_layers - 1;
    if intermediate_bounds.len() != expected_intermediate_bounds {
        return Err(MipError::Encoding(format!(
            "intermediate bounds layer count {} doesn't match {} implicit ReLU layers",
            intermediate_bounds.len(),
            expected_intermediate_bounds
        )));
    }

    let mut encoder = MipEncoder::new(input_bounds)?;

    for layer_idx in 0..num_layers {
        let out_dim = layer_dims[layer_idx + 1];
        encoder.encode_linear(&weights[layer_idx], &biases[layer_idx], out_dim)?;

        // Apply ReLU to all layers except the last
        if layer_idx < num_layers - 1 {
            encoder.encode_relu(&intermediate_bounds[layer_idx])?;
        }
    }

    encoder.finalize();
    Ok(encoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_holds(row: &crate::ir::RowSpec, values: &[f64]) -> bool {
        let value = row
            .coeffs
            .iter()
            .map(|&(column, coefficient)| coefficient * values[column])
            .sum::<f64>();
        value >= row.lb && value <= row.ub
    }

    #[test]
    fn f64_input_constructor_preserves_nonrepresentable_endpoints() {
        let lower = 0.1_f64;
        let upper = lower.next_up();
        assert_ne!(f64::from(lower as f32).to_bits(), lower.to_bits());

        let encoder = MipEncoder::new_with_f64_bounds(&[(lower, upper)])
            .expect("finite ordered binary64 box");
        let parts = encoder.into_parts();
        assert_eq!(parts.problem.cols()[0].lb.to_bits(), lower.to_bits());
        assert_eq!(parts.problem.cols()[0].ub.to_bits(), upper.to_bits());
    }

    #[test]
    fn f64_input_constructor_rejects_nan_and_inversion() {
        assert!(MipEncoder::new_with_f64_bounds(&[(f64::NAN, 1.0)]).is_err());
        assert!(MipEncoder::new_with_f64_bounds(&[(f64::NEG_INFINITY, 1.0)]).is_err());
        assert!(MipEncoder::new_with_f64_bounds(&[(0.0, f64::INFINITY)]).is_err());
        assert!(MipEncoder::new_with_f64_bounds(&[(f64::INFINITY, f64::INFINITY)]).is_err());
        assert!(MipEncoder::new_with_f64_bounds(&[(2.0, 1.0)]).is_err());
    }

    #[test]
    fn continuous_relu_outer_contains_graph_and_has_no_binary() {
        let mut encoder = MipEncoder::new_with_f64_bounds(&[(-2.0, 3.0)]).unwrap();
        encoder
            .encode_relu_continuous_outer(&[(-2.0, 3.0)])
            .unwrap();
        encoder.finalize();
        let parts = encoder.into_parts();

        assert!(parts.binary_vars.is_empty());
        assert_eq!(parts.problem.num_cols(), 2);
        assert_eq!(parts.problem.num_rows(), 2);
        assert_eq!(parts.problem.cols()[parts.output_vars[0].0].lb, 0.0);
        assert_eq!(parts.problem.cols()[parts.output_vars[0].0].ub, 3.0);

        for (x, y) in [(-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (3.0, 3.0)] {
            let values = [x, y];
            assert!(
                parts
                    .problem
                    .rows()
                    .iter()
                    .all(|row| row_holds(row, &values)),
                "ReLU graph point ({x}, {y}) must lie in its outer hull"
            );
        }
        assert!(
            parts
                .problem
                .rows()
                .iter()
                .all(|row| row_holds(row, &[0.0, 1.0])),
            "the relaxation must retain a genuine convex-hull interior point"
        );
        assert!(
            !parts
                .problem
                .rows()
                .iter()
                .all(|row| row_holds(row, &[0.0, 1.25])),
            "the Planet upper facet must exclude points above the hull"
        );
    }

    #[test]
    fn continuous_relu_outer_keeps_zero_boundary_phases_exact() {
        let mut encoder =
            MipEncoder::new_with_f64_bounds(&[(0.0, 2.0), (-2.0, 0.0), (0.0, 0.0)]).unwrap();
        encoder
            .encode_relu_continuous_outer(&[(0.0, 2.0), (-2.0, 0.0), (0.0, 0.0)])
            .unwrap();
        encoder.finalize();
        let parts = encoder.into_parts();

        assert_eq!(parts.problem.num_rows(), 0);
        assert_eq!(parts.problem.num_cols(), 4);
        assert_eq!(parts.output_vars[0], parts.input_vars[0]);
        assert_ne!(parts.output_vars[1], parts.input_vars[1]);
        assert_eq!(parts.problem.cols()[parts.output_vars[1].0].lb, 0.0);
        assert_eq!(parts.problem.cols()[parts.output_vars[1].0].ub, 0.0);
        assert_eq!(parts.output_vars[2], parts.input_vars[2]);
        assert!(parts.binary_vars.is_empty());
    }

    #[test]
    fn continuous_relu_outer_rejects_inexact_arithmetic_atomically() {
        let just_above_one = 1.0_f64.next_up();
        let mut inexact_product =
            MipEncoder::new_with_f64_bounds(&[(-1.0, 1.0), (-just_above_one, just_above_one)])
                .unwrap();
        let error = inexact_product
            .encode_relu_continuous_outer(&[(-1.0, 1.0), (-just_above_one, just_above_one)])
            .unwrap_err()
            .to_string();
        assert!(error.contains("product -u*l"), "unexpected error: {error}");
        let parts = inexact_product.into_parts();
        assert_eq!(parts.problem.num_cols(), 2);
        assert_eq!(parts.problem.num_rows(), 0);

        let mut inexact_width =
            MipEncoder::new_with_f64_bounds(&[(-f64::MAX, f64::MIN_POSITIVE)]).unwrap();
        let error = inexact_width
            .encode_relu_continuous_outer(&[(-f64::MAX, f64::MIN_POSITIVE)])
            .unwrap_err()
            .to_string();
        assert!(error.contains("width u-l"), "unexpected error: {error}");
        let parts = inexact_width.into_parts();
        assert_eq!(parts.problem.num_cols(), 1);
        assert_eq!(parts.problem.num_rows(), 0);
    }

    #[test]
    fn continuous_relu_outer_accepts_exact_binary32_hull_arithmetic() {
        let lower = f64::from(-0.1_f32);
        let upper = f64::from(0.2_f32);
        let mut encoder = MipEncoder::new_with_f64_bounds(&[(lower, upper)]).unwrap();
        encoder
            .encode_relu_continuous_outer(&[(lower, upper)])
            .expect("binary32 product and nearby difference fit exactly in binary64");
        assert_eq!(encoder.num_binary_vars(), 0);
    }
}
