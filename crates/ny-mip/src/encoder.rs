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
use crate::ir::{Col, MilpProblem};
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

    /// Mark the current frontier as output variables.
    pub fn finalize(&mut self) {
        self.output_vars = self.current_vars.clone();
    }

    /// Add constraint: output[i] <= output[j].
    ///
    /// Encodes as: Y_i - Y_j <= 0.
    /// Used for VNNLIB LessEq(i, j) constraints.
    pub fn constrain_output_leq(&mut self, i: usize, j: usize) -> Result<()> {
        let (yi, yj) = self.output_pair(i, j)?;
        self.problem
            .add_row(f64::NEG_INFINITY, 0.0, [(yi, 1.0), (yj, -1.0)]);
        Ok(())
    }

    /// Add constraint: output[i] >= output[j].
    ///
    /// Encodes as: Y_i - Y_j >= 0.
    /// Used for VNNLIB GreaterEq(i, j) constraints.
    pub fn constrain_output_geq(&mut self, i: usize, j: usize) -> Result<()> {
        let (yi, yj) = self.output_pair(i, j)?;
        self.problem
            .add_row(0.0, f64::INFINITY, [(yi, 1.0), (yj, -1.0)]);
        Ok(())
    }

    /// Add constraint: output[i] <= constant.
    ///
    /// Used for VNNLIB LessEqConst(i, c) constraints.
    pub fn constrain_output_leq_const(&mut self, i: usize, val: f64) -> Result<()> {
        let yi = self.output_var(i)?;
        self.problem.add_row(f64::NEG_INFINITY, val, [(yi, 1.0)]);
        Ok(())
    }

    /// Add constraint: output[i] >= constant.
    ///
    /// Used for VNNLIB GreaterEqConst(i, c) constraints.
    pub fn constrain_output_geq_const(&mut self, i: usize, val: f64) -> Result<()> {
        let yi = self.output_var(i)?;
        self.problem.add_row(val, f64::INFINITY, [(yi, 1.0)]);
        Ok(())
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
