// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! V2 (ENGINE_PROGRAM §2 V, `#vit-root-alpha`): the plain-data GRAPH search
//! copy and the vit root α search request.
//!
//! The margin-row lane's search copy ([`super::SearchNet`]) is a straight-line
//! conv/affine/ReLU net. The vit lane's IR is a `GraphNetwork` DAG whose root
//! relaxation carries three optimisable families:
//!
//! * the ReLU lower slopes of the MLP blocks;
//! * the McCormick interpolation parameters of every var×var attention product
//!   (`Q·Kᵀ` and `attn·V`, `[4, m, n, k]` per product, shared over the batch
//!   (head) dims — the lane's `BilinearCrown` α layout, bit for bit);
//! * the McCormick interpolation parameters of the fixed-index softmax's
//!   `exp × 1/Σexp` product (`[2, n]` per MulBinary — the lane's MulBinary α
//!   layout), which `#vit-margin-alpha-complex` held frozen.
//!
//! [`GraphSearchNet`] is a LOSSLESS-ENOUGH plain-data copy for SEARCH: every
//! exact-linear node is an explicit affine map ([`GraphSearchOp::Sparse`],
//! [`GraphSearchOp::Dense`], [`GraphSearchOp::Conv`]), every relaxed node keeps
//! its operands, and the monolithic softmax carries the lane's own frozen LSE
//! affine bounds per group. Nothing certified reads it (R3): the lane scores a
//! proposal with its own certified backward over its own IR, so a lowering bug
//! costs tightness or a refusal, never soundness. The lane also self-checks the
//! copy against its own point forward before searching on it (G1).
//!
//! Every search output is [`Untrusted`]; the only product is a
//! [`GraphAlphaHint`], consumed by the lane's ingest gate (finite, clamped into
//! `[0, 1]`, one certified evaluation, kept only when strictly better).

use std::time::Instant;

use super::{SearchConv, Untrusted, GATHER_PAD};

/// One term of a [`GraphSearchOp::Sparse`] map:
/// `out[out] += weight * inputs[slot][index]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseTerm {
    /// Output element.
    pub out: u32,
    /// Position in the op's `inputs` list.
    pub slot: u32,
    /// Element of that input tensor.
    pub index: u32,
    /// Coefficient.
    pub weight: f64,
}

/// An elementwise nonlinearity the search relaxes with FROZEN lines (no
/// optimisable parameter).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryKind {
    /// `exp(x)`.
    Exp,
    /// `1 / x` (the search requires a strictly positive box).
    Reciprocal,
    /// `min(max(x, lo), hi)`.
    Clip {
        /// Lower clamp.
        lo: f64,
        /// Upper clamp.
        hi: f64,
    },
}

impl UnaryKind {
    /// The function itself, in f64 (reference forward only).
    #[must_use]
    pub fn eval(self, x: f64) -> f64 {
        match self {
            Self::Exp => x.exp(),
            Self::Reciprocal => 1.0 / x,
            Self::Clip { lo, hi } => x.max(lo).min(hi),
        }
    }

    /// Stable token for telemetry.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Reciprocal => "reciprocal",
            Self::Clip { .. } => "clip",
        }
    }
}

/// One op of a [`GraphSearchNet`]; tensor `k + 1` is the output of op `k`,
/// tensor 0 is the network input. Every tensor is flat, row-major.
#[derive(Debug, Clone)]
pub enum GraphSearchOp {
    /// Any exact affine map over one or more earlier tensors:
    /// `out[o] = bias[o] + Σ weight · inputs[slot][index]` (reshapes,
    /// transposes, concats, broadcasts, batch norms, constant arithmetic,
    /// residual adds, reductions).
    Sparse {
        /// Consumed tensor ids.
        inputs: Vec<usize>,
        /// Output width.
        n_out: usize,
        /// The nonzero coefficients.
        terms: Vec<SparseTerm>,
        /// `[n_out]` constant.
        bias: Vec<f64>,
    },
    /// A last-axis linear layer: the input is `[rows][n_in]`, the output
    /// `[rows][n_out]`, `out[r][o] = bias[o] + Σ_i weight[o][i] · in[r][i]`.
    Dense {
        /// Consumed tensor id.
        input: usize,
        /// Leading positions (tokens).
        rows: usize,
        /// Input features.
        n_in: usize,
        /// Output features.
        n_out: usize,
        /// Row-major `[n_out][n_in]` weights.
        weight: Vec<f64>,
        /// `[n_out]` bias.
        bias: Vec<f64>,
    },
    /// A convolution in gather form (see [`SearchConv`]).
    Conv(Box<SearchConv>),
    /// ReLU (an α site: one lower slope per neuron).
    Relu {
        /// Consumed tensor id.
        input: usize,
    },
    /// A var×var product `out[b][i][j] = scale · Σ_l A[b][i][l] · B(b, l, j)`
    /// with `A` flat `[batch][m][k]` and `B` flat `[batch][k][n]` (or
    /// `[batch][n][k]` when `transpose_b`). An α site: `[4][m][n][k]`
    /// interpolation parameters shared over `batch` — `[0]` / `[2]` are the
    /// lower-direction lower / upper planes, `[1]` / `[3]` the upper-direction
    /// ones (the lane's `BilinearCrown` layout).
    Bilinear {
        /// Left operand tensor id.
        a: usize,
        /// Right operand tensor id.
        b: usize,
        /// Batch (head) count.
        batch: usize,
        /// Rows of `A`.
        m: usize,
        /// Output columns.
        n: usize,
        /// Contraction length.
        k: usize,
        /// `B` is stored `[n][k]` per batch.
        transpose_b: bool,
        /// Output scale.
        scale: f64,
    },
    /// Elementwise product of two equally sized tensors (an α site: `[2][n]`
    /// interpolation parameters, `[0]` the lower plane, `[1]` the upper one —
    /// the lane's MulBinary layout).
    MulBinary {
        /// Left operand tensor id.
        a: usize,
        /// Right operand tensor id.
        b: usize,
    },
    /// An elementwise nonlinearity with frozen search lines.
    Unary {
        /// Consumed tensor id.
        input: usize,
        /// Which function.
        kind: UnaryKind,
    },
    /// Last-axis softmax over `groups` groups of `len`, with the lane's own
    /// FROZEN affine bounds per group: output `(g, i)` lies between
    /// `lower_a[g][i][:] · in[g][:] + lower_b[g][i]` and the `upper_*` twin.
    Softmax {
        /// Consumed tensor id.
        input: usize,
        /// Groups.
        groups: usize,
        /// Group length.
        len: usize,
        /// `[groups][len][len]` lower coefficients.
        lower_a: Vec<f64>,
        /// `[groups][len]` lower constants.
        lower_b: Vec<f64>,
        /// `[groups][len][len]` upper coefficients.
        upper_a: Vec<f64>,
        /// `[groups][len]` upper constants.
        upper_b: Vec<f64>,
    },
}

/// The plain-data search copy of a graph lane's network (SEARCH only).
#[derive(Debug, Clone)]
pub struct GraphSearchNet {
    /// Flat input size (tensor 0).
    pub n_in: usize,
    /// Ops in execution order; the LAST op's output is the objective tensor.
    pub ops: Vec<GraphSearchOp>,
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

impl GraphSearchNet {
    /// The objective tensor (the last op's output).
    #[must_use]
    pub fn output(&self) -> usize {
        self.ops.len()
    }

    /// Validate the op list and return every tensor's flat size (index 0 is
    /// the input). Fails on a forward reference, a size mismatch, an index out
    /// of range or a non-finite parameter.
    pub fn tensor_sizes(&self) -> Result<Vec<usize>, String> {
        if self.n_in == 0 || self.ops.is_empty() {
            return Err("empty graph search net".into());
        }
        let mut sizes = vec![self.n_in];
        for (k, op) in self.ops.iter().enumerate() {
            let size_of = |id: usize| -> Result<usize, String> {
                sizes
                    .get(id)
                    .copied()
                    .ok_or_else(|| format!("op {k} references tensor {id} before it exists"))
            };
            let out = match op {
                GraphSearchOp::Sparse {
                    inputs,
                    n_out,
                    terms,
                    bias,
                } => {
                    if inputs.is_empty() || *n_out == 0 || bias.len() != *n_out {
                        return Err(format!("sparse {k}: empty or inconsistent shape"));
                    }
                    let widths = inputs
                        .iter()
                        .map(|&id| size_of(id))
                        .collect::<Result<Vec<usize>, String>>()?;
                    for term in terms {
                        let slot = term.slot as usize;
                        if term.out as usize >= *n_out
                            || slot >= widths.len()
                            || term.index as usize >= widths[slot]
                            || !term.weight.is_finite()
                        {
                            return Err(format!("sparse {k}: term out of range or non-finite"));
                        }
                    }
                    if !all_finite(bias) {
                        return Err(format!("sparse {k}: non-finite bias"));
                    }
                    *n_out
                }
                GraphSearchOp::Dense {
                    input,
                    rows,
                    n_in,
                    n_out,
                    weight,
                    bias,
                } => {
                    let n = size_of(*input)?;
                    if *rows == 0
                        || *n_in == 0
                        || *n_out == 0
                        || rows.checked_mul(*n_in) != Some(n)
                        || weight.len() != n_out * n_in
                        || bias.len() != *n_out
                    {
                        return Err(format!("dense {k}: shape mismatch"));
                    }
                    if !all_finite(weight) || !all_finite(bias) {
                        return Err(format!("dense {k}: non-finite parameters"));
                    }
                    rows * n_out
                }
                GraphSearchOp::Conv(c) => {
                    let n_src = size_of(c.input)?;
                    if c.cout == 0
                        || c.k == 0
                        || c.p == 0
                        || c.gather.len() != c.k * c.p
                        || c.weight.len() != c.cout * c.k
                        || c.bias.len() != c.cout
                    {
                        return Err(format!("conv {k}: inconsistent table sizes"));
                    }
                    if c.gather
                        .iter()
                        .any(|&g| g != GATHER_PAD && g as usize >= n_src)
                    {
                        return Err(format!("conv {k}: gather index out of range"));
                    }
                    if !all_finite(&c.weight) || !all_finite(&c.bias) {
                        return Err(format!("conv {k}: non-finite parameters"));
                    }
                    c.cout * c.p
                }
                GraphSearchOp::Relu { input } => size_of(*input)?,
                GraphSearchOp::Bilinear {
                    a,
                    b,
                    batch,
                    m,
                    n,
                    k: kk,
                    scale,
                    ..
                } => {
                    let (na, nb) = (size_of(*a)?, size_of(*b)?);
                    if *batch == 0 || *m == 0 || *n == 0 || *kk == 0 || !scale.is_finite() {
                        return Err(format!("bilinear {k}: empty dimension or bad scale"));
                    }
                    if Some(na) != batch.checked_mul(*m).and_then(|v| v.checked_mul(*kk))
                        || Some(nb) != batch.checked_mul(*kk).and_then(|v| v.checked_mul(*n))
                    {
                        return Err(format!("bilinear {k}: operand sizes disagree with shape"));
                    }
                    batch * m * n
                }
                GraphSearchOp::MulBinary { a, b } => {
                    let (na, nb) = (size_of(*a)?, size_of(*b)?);
                    if na != nb {
                        return Err(format!("mul {k}: operand sizes {na} vs {nb}"));
                    }
                    na
                }
                GraphSearchOp::Unary { input, kind } => {
                    if let UnaryKind::Clip { lo, hi } = kind {
                        if lo.is_nan() || hi.is_nan() || lo > hi {
                            return Err(format!("clip {k}: bad bounds"));
                        }
                    }
                    size_of(*input)?
                }
                GraphSearchOp::Softmax {
                    input,
                    groups,
                    len,
                    lower_a,
                    lower_b,
                    upper_a,
                    upper_b,
                } => {
                    let n = size_of(*input)?;
                    let cells = groups.checked_mul(*len).and_then(|v| v.checked_mul(*len));
                    if *groups == 0
                        || *len == 0
                        || groups.checked_mul(*len) != Some(n)
                        || cells != Some(lower_a.len())
                        || cells != Some(upper_a.len())
                        || lower_b.len() != n
                        || upper_b.len() != n
                    {
                        return Err(format!("softmax {k}: shape mismatch"));
                    }
                    if !all_finite(lower_a)
                        || !all_finite(lower_b)
                        || !all_finite(upper_a)
                        || !all_finite(upper_b)
                    {
                        return Err(format!("softmax {k}: non-finite relaxation"));
                    }
                    n
                }
            };
            sizes.push(out);
        }
        Ok(sizes)
    }

    /// Plain f64 point evaluation of every tensor (index 0 is `x`), with the
    /// TRUE functions at relaxed ops. A REFERENCE for the lane's lowering
    /// self-check (G1) and for tests; nothing verdict-bearing reads it.
    pub fn forward_f64(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        let sizes = self.tensor_sizes()?;
        if x.len() != self.n_in {
            return Err(format!(
                "input has {} values, net expects {}",
                x.len(),
                self.n_in
            ));
        }
        let mut vals: Vec<Vec<f64>> = Vec::with_capacity(sizes.len());
        vals.push(x.to_vec());
        for op in &self.ops {
            let out = match op {
                GraphSearchOp::Sparse {
                    inputs,
                    terms,
                    bias,
                    ..
                } => {
                    let mut dst = bias.clone();
                    for term in terms {
                        let src = &vals[inputs[term.slot as usize]];
                        dst[term.out as usize] += term.weight * src[term.index as usize];
                    }
                    dst
                }
                GraphSearchOp::Dense {
                    input,
                    rows,
                    n_in,
                    n_out,
                    weight,
                    bias,
                } => {
                    let src = &vals[*input];
                    let mut dst = vec![0.0; rows * n_out];
                    for r in 0..*rows {
                        for o in 0..*n_out {
                            let mut acc = bias[o];
                            for i in 0..*n_in {
                                acc += weight[o * n_in + i] * src[r * n_in + i];
                            }
                            dst[r * n_out + o] = acc;
                        }
                    }
                    dst
                }
                GraphSearchOp::Conv(c) => {
                    let src = &vals[c.input];
                    let mut dst = vec![0.0; c.cout * c.p];
                    for o in 0..c.cout {
                        for sp in 0..c.p {
                            let mut acc = c.bias[o];
                            for t in 0..c.k {
                                let g = c.gather[t * c.p + sp];
                                if g != GATHER_PAD {
                                    acc += c.weight[o * c.k + t] * src[g as usize];
                                }
                            }
                            dst[o * c.p + sp] = acc;
                        }
                    }
                    dst
                }
                GraphSearchOp::Relu { input } => vals[*input].iter().map(|v| v.max(0.0)).collect(),
                GraphSearchOp::Bilinear {
                    a,
                    b,
                    batch,
                    m,
                    n,
                    k,
                    transpose_b,
                    scale,
                } => {
                    let (va, vb) = (&vals[*a], &vals[*b]);
                    let mut dst = vec![0.0; batch * m * n];
                    for bb in 0..*batch {
                        for i in 0..*m {
                            for j in 0..*n {
                                let mut acc = 0.0;
                                for l in 0..*k {
                                    let x = va[(bb * m + i) * k + l];
                                    let y = if *transpose_b {
                                        vb[(bb * n + j) * k + l]
                                    } else {
                                        vb[(bb * k + l) * n + j]
                                    };
                                    acc += x * y;
                                }
                                dst[(bb * m + i) * n + j] = scale * acc;
                            }
                        }
                    }
                    dst
                }
                GraphSearchOp::MulBinary { a, b } => {
                    vals[*a].iter().zip(&vals[*b]).map(|(x, y)| x * y).collect()
                }
                GraphSearchOp::Unary { input, kind } => {
                    vals[*input].iter().map(|&v| kind.eval(v)).collect()
                }
                GraphSearchOp::Softmax {
                    input, groups, len, ..
                } => {
                    let src = &vals[*input];
                    let mut dst = vec![0.0; groups * len];
                    for g in 0..*groups {
                        let row = &src[g * len..(g + 1) * len];
                        let shift = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let exps: Vec<f64> = row.iter().map(|v| (v - shift).exp()).collect();
                        let sum: f64 = exps.iter().sum();
                        for (i, e) in exps.iter().enumerate() {
                            dst[g * len + i] = e / sum;
                        }
                    }
                    dst
                }
            };
            vals.push(out);
        }
        Ok(vals)
    }

    /// Approximate resident bytes of this copy (admission accounting).
    #[must_use]
    pub fn approx_bytes(&self) -> u64 {
        let f = size_of::<f64>() as u64;
        self.ops
            .iter()
            .map(|op| match op {
                GraphSearchOp::Sparse {
                    inputs,
                    terms,
                    bias,
                    ..
                } => {
                    (terms.len() as u64) * (size_of::<SparseTerm>() as u64)
                        + ((inputs.len() + bias.len()) as u64) * f
                }
                GraphSearchOp::Dense { weight, bias, .. } => {
                    ((weight.len() + bias.len()) as u64) * f
                }
                GraphSearchOp::Conv(c) => {
                    (c.gather.len() as u64) * 4 + ((c.weight.len() + c.bias.len()) as u64) * f
                }
                GraphSearchOp::Softmax {
                    lower_a,
                    lower_b,
                    upper_a,
                    upper_b,
                    ..
                } => ((lower_a.len() + lower_b.len() + upper_a.len() + upper_b.len()) as u64) * f,
                GraphSearchOp::Relu { .. }
                | GraphSearchOp::Bilinear { .. }
                | GraphSearchOp::MulBinary { .. }
                | GraphSearchOp::Unary { .. } => 0,
            })
            .fold(0u64, u64::saturating_add)
    }
}

/// A tensor's REFERENCE box (the lane's certified box). The search relaxes
/// every nonlinear op over the boxes of its operands; nothing flows back.
#[derive(Debug, Clone)]
pub struct TensorBox {
    /// Tensor id in the [`GraphSearchNet`].
    pub tensor: usize,
    /// Lower sides.
    pub l: Vec<f64>,
    /// Upper sides.
    pub u: Vec<f64>,
}

/// The V2 hint: α per site, keyed by OP index in the [`GraphSearchNet`].
///
/// * `relu`: one lower slope per neuron of the ReLU's input;
/// * `bilinear`: the full `[4][m][n][k]` parameter block;
/// * `mul_binary`: the full `[2][n]` parameter block.
///
/// The ingest gate reads only entries whose length matches the lane's own
/// state, refuses every non-finite value (the incumbent stays) and clamps the
/// rest into `[0, 1]`.
#[derive(Debug, Clone, Default)]
pub struct GraphAlphaHint {
    /// `(op, slopes)` per ReLU site.
    pub relu: Vec<(usize, Vec<f32>)>,
    /// `(op, [4 m n k] parameters)` per bilinear site.
    pub bilinear: Vec<(usize, Vec<f32>)>,
    /// `(op, [2 n] parameters)` per MulBinary site.
    pub mul_binary: Vec<(usize, Vec<f32>)>,
}

/// One V2 request: raise the LOWER bounds of the objective rows
/// `rows[r] · y` (y = the last tensor) over the input box `[lo, hi]`; row `r`
/// counts as verified in the search when its bound exceeds `thresholds[r]`.
#[derive(Debug, Clone)]
pub struct GraphRootRequest {
    /// Input box lower sides.
    pub lo: Vec<f64>,
    /// Input box upper sides.
    pub hi: Vec<f64>,
    /// Reference boxes: at least every operand of every nonlinear op.
    pub boxes: Vec<TensorBox>,
    /// Objective rows over the last tensor (LOWER objectives).
    pub rows: Vec<Vec<f64>>,
    /// Per-row thresholds.
    pub thresholds: Vec<f64>,
    /// The incumbent's α (the search starts here; missing sites start at
    /// the lane's defaults).
    pub init: GraphAlphaHint,
    /// Iterations after the initial evaluation.
    pub iters: usize,
    /// Adam step.
    pub lr: f64,
    /// Wall deadline; the search stops at it and returns its best so far.
    pub deadline: Option<Instant>,
    /// Worker threads (0 = the backend's choice).
    pub threads: usize,
    /// Admitted working-set bytes.
    pub mem_cap_bytes: u64,
}

/// Telemetry of one V2 search. Every bound here is an UNTRUSTED search number
/// (f32, no rounding charged) and is never published.
#[derive(Debug)]
pub struct GraphRootStats {
    /// Iterations completed after the initial evaluation.
    pub iters: usize,
    /// Full (all-row) evaluations.
    pub evals: usize,
    /// Wall seconds of lowering + search.
    pub search_secs: f64,
    /// Worst row slack (`bound - threshold`) at the incumbent's α.
    pub init_slack: Untrusted<f64>,
    /// Worst row slack at the returned α.
    pub best_slack: Untrusted<f64>,
    /// Objective rows.
    pub rows: usize,
    /// Rows the search saw verified at the incumbent's α.
    pub verified_init: usize,
    /// Rows the search saw verified at the returned α.
    pub verified_best: usize,
    /// ReLU / bilinear / MulBinary sites searched.
    pub relu_sites: usize,
    /// Bilinear sites searched.
    pub bilinear_sites: usize,
    /// MulBinary sites searched.
    pub mul_sites: usize,
    /// Worker threads used.
    pub threads: usize,
    /// Why the search stopped: `verified`, `patience`, `iters` or `deadline`.
    pub stop: &'static str,
}

/// What `SearchBackend::graph_root_alpha` returns.
#[derive(Debug)]
pub struct GraphRootOutcome {
    /// The parameters, for the ingest gate only.
    pub alpha: Untrusted<GraphAlphaHint>,
    /// Telemetry.
    pub stats: GraphRootStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-token attention-shaped head: a per-token dense layer, a ReLU,
    /// `q·qᵀ`, a softmax (frozen constant bounds), a sparse mix and a clip.
    fn head() -> GraphSearchNet {
        GraphSearchNet {
            n_in: 4,
            ops: vec![
                // t1 = x viewed [2][2] -> dense 2->2 per row
                GraphSearchOp::Dense {
                    input: 0,
                    rows: 2,
                    n_in: 2,
                    n_out: 2,
                    weight: vec![1.0, -1.0, 0.5, 2.0],
                    bias: vec![0.0, 0.25],
                },
                // t2 = relu(t1)
                GraphSearchOp::Relu { input: 1 },
                // t3 = t2 · t2ᵀ (batch 1, m = 2, k = 2, n = 2)
                GraphSearchOp::Bilinear {
                    a: 2,
                    b: 2,
                    batch: 1,
                    m: 2,
                    n: 2,
                    k: 2,
                    transpose_b: true,
                    scale: 0.5,
                },
                // t4 = softmax over rows (constant frozen bounds are enough here)
                GraphSearchOp::Softmax {
                    input: 3,
                    groups: 2,
                    len: 2,
                    lower_a: vec![0.0; 8],
                    lower_b: vec![0.0; 4],
                    upper_a: vec![0.0; 8],
                    upper_b: vec![1.0; 4],
                },
                // t5 = [t4[0] + t4[3], t4[1] - 2 t4[2]] + 0.5
                GraphSearchOp::Sparse {
                    inputs: vec![4],
                    n_out: 2,
                    terms: vec![
                        SparseTerm {
                            out: 0,
                            slot: 0,
                            index: 0,
                            weight: 1.0,
                        },
                        SparseTerm {
                            out: 0,
                            slot: 0,
                            index: 3,
                            weight: 1.0,
                        },
                        SparseTerm {
                            out: 1,
                            slot: 0,
                            index: 1,
                            weight: 1.0,
                        },
                        SparseTerm {
                            out: 1,
                            slot: 0,
                            index: 2,
                            weight: -2.0,
                        },
                    ],
                    bias: vec![0.5, 0.5],
                },
                // t6 = clip(t5, 0, 1)
                GraphSearchOp::Unary {
                    input: 5,
                    kind: UnaryKind::Clip { lo: 0.0, hi: 1.0 },
                },
            ],
        }
    }

    #[test]
    fn graph_search_net_sizes_and_forward_are_exact_on_an_attention_head() {
        let net = head();
        assert_eq!(
            net.tensor_sizes().expect("valid"),
            vec![4, 4, 4, 4, 4, 2, 2]
        );
        assert_eq!(net.output(), 6);
        let vals = net.forward_f64(&[1.0, 0.5, -1.0, 2.0]).expect("forward");
        // Dense per row: row0 = [1 - 0.5, 0.5 + 1 + 0.25] = [0.5, 1.75],
        // row1 = [-1 - 2, -0.5 + 4 + 0.25] = [-3, 3.75].
        assert_eq!(vals[1], vec![0.5, 1.75, -3.0, 3.75]);
        assert_eq!(vals[2], vec![0.5, 1.75, 0.0, 3.75]);
        // q·qᵀ · 0.5: [[0.25+3.0625, 6.5625], [6.5625, 14.0625]] · 0.5
        let expect = [
            f64::midpoint(0.25, 1.75 * 1.75),
            0.5 * (1.75 * 3.75),
            0.5 * (1.75 * 3.75),
            0.5 * (3.75 * 3.75),
        ];
        for (got, want) in vals[3].iter().zip(expect) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        // Softmax rows sum to one.
        assert!((vals[4][0] + vals[4][1] - 1.0).abs() < 1e-12);
        assert!((vals[4][2] + vals[4][3] - 1.0).abs() < 1e-12);
        // Same association as the forward: bias first, then terms in order.
        let t5 = [
            (0.5 + vals[4][0]) + vals[4][3],
            (0.5 + vals[4][1]) + (-2.0 * vals[4][2]),
        ];
        assert_eq!(vals[5], t5.to_vec());
        assert_eq!(
            vals[6],
            t5.iter().map(|v| v.clamp(0.0, 1.0)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn graph_search_net_validation_refuses_bad_shapes_and_values() {
        let mut net = head();
        net.ops[2] = GraphSearchOp::Bilinear {
            a: 2,
            b: 2,
            batch: 1,
            m: 3,
            n: 2,
            k: 2,
            transpose_b: true,
            scale: 0.5,
        };
        assert!(net.tensor_sizes().is_err(), "m*k must equal |A|");
        let mut net = head();
        if let GraphSearchOp::Sparse { terms, .. } = &mut net.ops[4] {
            terms[0].index = 4;
        }
        assert!(net.tensor_sizes().is_err(), "an index past the input");
        let mut net = head();
        if let GraphSearchOp::Softmax { upper_b, .. } = &mut net.ops[3] {
            upper_b[1] = f64::INFINITY;
        }
        assert!(
            net.tensor_sizes().is_err(),
            "a non-finite frozen relaxation"
        );
        let mut net = head();
        net.ops[0] = GraphSearchOp::Relu { input: 3 };
        assert!(net.tensor_sizes().is_err(), "a forward reference");
        assert!(GraphSearchNet {
            n_in: 0,
            ops: vec![]
        }
        .tensor_sizes()
        .is_err());
        assert!(head().approx_bytes() > 0);
    }

    #[test]
    fn unary_kinds_evaluate_their_functions() {
        assert_eq!(UnaryKind::Exp.eval(0.0), 1.0);
        assert_eq!(UnaryKind::Reciprocal.eval(4.0), 0.25);
        let clip = UnaryKind::Clip { lo: -1.0, hi: 2.0 };
        assert_eq!(clip.eval(-3.0), -1.0);
        assert_eq!(clip.eval(0.5), 0.5);
        assert_eq!(clip.eval(7.0), 2.0);
        assert_eq!(clip.token(), "clip");
    }
}
