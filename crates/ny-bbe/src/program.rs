// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `BoundProgram` lowering (ENGINE_PROGRAM §1.3): a [`SearchNet`] (the lane's
//! plain-data IR copy, f64) becomes the search's own program in the search
//! scalar, with the conv gather tables transposed to `[p][k]` so one output
//! position's taps are contiguous.
//!
//! UNTRUSTED: nothing certified ever reads a [`Program`]. A lowering bug costs
//! tightness (a worse hint) or a refusal; the lane's certified evaluator reads
//! the lane's own IR (R3).

use ny_core::search::{SearchNet, SearchOp, GATHER_PAD};

use crate::real::Real;

/// Padding / "no position" sentinel.
pub(crate) const NONE: u32 = u32::MAX;

/// One lowered op; tensor `k + 1` is the output of op `k`.
pub(crate) enum Op<R> {
    /// Convolution, `[p][k]` gather, `[cout][k]` weights.
    Conv {
        input: usize,
        k: usize,
        p: usize,
        gather_pk: Vec<u32>,
        w: Vec<R>,
        b: Vec<R>,
    },
    /// Per-neuron affine.
    Affine {
        input: usize,
        scale: Vec<R>,
        shift: Vec<R>,
    },
    /// ReLU at search site `site`.
    Relu { input: usize, site: usize },
    /// Elementwise sum.
    Add { lhs: usize, rhs: usize },
    /// Flat identity.
    Copy { input: usize },
    /// Dense layer, row-major `[n_out][n_in]`.
    Gemm {
        input: usize,
        n_in: usize,
        w: Vec<R>,
        b: Vec<R>,
    },
}

/// One ReLU of the program: a START NODE of the joint search (its
/// pre-activation rows) and an owner of slopes for every later start node.
pub(crate) struct Site {
    /// The ReLU's op index.
    pub(crate) op: usize,
    /// Its pre-activation tensor id.
    pub(crate) input: usize,
    /// Width.
    pub(crate) n: usize,
    /// The lane's head ReLU.
    pub(crate) head: bool,
}

/// The lowered search program.
pub(crate) struct Program<R> {
    /// Input width.
    pub(crate) n_in: usize,
    /// Flat size of every tensor (0 = input).
    pub(crate) tsize: Vec<usize>,
    /// Spatial positions per channel of every tensor (1 when unknown); used
    /// only to GROUP rows whose receptive fields coincide into one block.
    pub(crate) spatial: Vec<usize>,
    /// Ops in execution order.
    pub(crate) ops: Vec<Op<R>>,
    /// ReLU sites in op order.
    pub(crate) sites: Vec<Site>,
}

impl<R: Real> Program<R> {
    /// Lower a validated [`SearchNet`] into the search scalar.
    pub(crate) fn lower(net: &SearchNet) -> Result<Self, String> {
        let tsize = net.tensor_sizes()?;
        // Neuron indices travel as u32 with `NONE` as the sentinel.
        if tsize.iter().any(|&n| n >= NONE as usize) {
            return Err("a tensor is too wide for u32 neuron indices".into());
        }
        let mut spatial = vec![1usize; tsize.len()];
        let mut ops = Vec::with_capacity(net.ops.len());
        let mut sites = Vec::new();
        let cvt = |v: &[f64]| -> Vec<R> { v.iter().map(|&x| R::from_f64(x)).collect() };
        for (k, op) in net.ops.iter().enumerate() {
            let out = k + 1;
            let lowered = match op {
                SearchOp::Conv(c) => {
                    let mut gather_pk = vec![NONE; c.k * c.p];
                    for t in 0..c.k {
                        for sp in 0..c.p {
                            let g = c.gather[t * c.p + sp];
                            gather_pk[sp * c.k + t] = if g == GATHER_PAD { NONE } else { g };
                        }
                    }
                    spatial[out] = c.p;
                    Op::Conv {
                        input: c.input,
                        k: c.k,
                        p: c.p,
                        gather_pk,
                        w: cvt(&c.weight),
                        b: cvt(&c.bias),
                    }
                }
                SearchOp::Affine {
                    input,
                    scale,
                    shift,
                } => {
                    spatial[out] = spatial[*input];
                    Op::Affine {
                        input: *input,
                        scale: cvt(scale),
                        shift: cvt(shift),
                    }
                }
                SearchOp::Relu { input, head } => {
                    spatial[out] = spatial[*input];
                    sites.push(Site {
                        op: k,
                        input: *input,
                        n: tsize[*input],
                        head: *head,
                    });
                    Op::Relu {
                        input: *input,
                        site: sites.len() - 1,
                    }
                }
                SearchOp::Add { lhs, rhs } => {
                    spatial[out] = spatial[*lhs];
                    Op::Add {
                        lhs: *lhs,
                        rhs: *rhs,
                    }
                }
                SearchOp::Flatten { input } => Op::Copy { input: *input },
                SearchOp::Gemm {
                    input,
                    n_in,
                    weight,
                    bias,
                    ..
                } => Op::Gemm {
                    input: *input,
                    n_in: *n_in,
                    w: cvt(weight),
                    b: cvt(bias),
                },
            };
            ops.push(lowered);
        }
        Ok(Self {
            n_in: net.n_in,
            tsize,
            spatial,
            ops,
            sites,
        })
    }

    /// The objective tensor (the last op's output).
    pub(crate) fn out_tensor(&self) -> usize {
        self.ops.len()
    }

    /// Bytes of one worker's workspace for a start node owning `stride`
    /// slopes: every tensor's `LANES`-wide buffer plus its mark and touched
    /// list, the dense per-slope Lambda rows and one worst-case block's
    /// sparse gradient (`stride` entries of a `u32` plus `lanes` values),
    /// which is what lets a gradient chunk never drop below the worker count.
    pub(crate) fn workspace_bytes(&self, lanes: usize, stride: usize) -> u64 {
        let per_neuron = (lanes * R::BYTES + 2 * 4) as u64;
        let tensors: u64 = self
            .tsize
            .iter()
            .map(|&n| n as u64 * per_neuron)
            .fold(0u64, u64::saturating_add);
        let entry = (4 + lanes * R::BYTES).next_multiple_of(R::BYTES.max(4));
        let rows = (stride as u64).saturating_mul((lanes * R::BYTES + entry) as u64);
        tensors.saturating_add(rows)
    }
}
