// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Plain-data description of a twin-wall resnet (#twinwall).
//!
//! Built by the caller (the ny-cli ONNX extractor, which owns BN folding in
//! f64) and compiled by [`super::net::TwinNet`]. ny-propagate deliberately has
//! no ONNX dependency, so this is the crate boundary: a validated, f64,
//! shape-explicit op list for the cifar100/tinyimagenet resnet family
//! (Conv+BN-folded trunk with ReLU/Add, head `Gemm -> ReLU -> Gemm`).

/// One trunk/head operation. Tensor ids: 0 is the network input; op `k`
/// produces tensor `k + 1`. `input`/`lhs`/`rhs` reference tensor ids.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // Vec-heavy Conv payload; specs are few and short-lived
pub enum TwinOpSpec {
    /// Folded convolution (BN already multiplied in, in f64, by the builder).
    Conv {
        /// Producer tensor id consumed.
        input: usize,
        /// Kernel `[cout][cin][kh][kw]`, row-major flattened.
        weight: Vec<f64>,
        /// Per-output-channel bias (BN shift folded in).
        bias: Vec<f64>,
        /// Certified ABSOLUTE per-channel error bound on `bias` vs the exact
        /// real-arithmetic fold (cancellation-safe; from the builder).
        bias_err: Vec<f64>,
        /// Certified RELATIVE error bound on every `weight` entry vs the
        /// exact real fold (0.0 when no BN was folded).
        weight_rel_err: f64,
        /// (cout, cin, kh, kw).
        kernel: (usize, usize, usize, usize),
        /// (stride_h, stride_w).
        stride: (usize, usize),
        /// (pad_top, pad_left, pad_bottom, pad_right).
        pads: (usize, usize, usize, usize),
        /// Input (C, H, W).
        ishape: (usize, usize, usize),
        /// Output (C, H, W).
        oshape: (usize, usize, usize),
    },
    /// Transposed convolution (BN foldable like Conv). Weights are given in
    /// the SAME `[cout][cin][kh][kw]` layout as `Conv` (the extractor
    /// transposes ONNX's `[cin][cout][kh][kw]` ConvTranspose layout); the
    /// compiler builds transpose-aware gather tables into the same conv
    /// kernel machinery (#epoch-bab Phase D).
    ConvTranspose {
        /// Producer tensor id consumed.
        input: usize,
        /// Kernel `[cout][cin][kh][kw]`, row-major flattened.
        weight: Vec<f64>,
        /// Per-output-channel bias (BN shift folded in).
        bias: Vec<f64>,
        /// Certified ABSOLUTE per-channel error bound on `bias`.
        bias_err: Vec<f64>,
        /// Certified RELATIVE error bound on every `weight` entry.
        weight_rel_err: f64,
        /// (cout, cin, kh, kw).
        kernel: (usize, usize, usize, usize),
        /// (stride_h, stride_w).
        stride: (usize, usize),
        /// (pad_top, pad_left, pad_bottom, pad_right).
        pads: (usize, usize, usize, usize),
        /// Input (C, H, W).
        ishape: (usize, usize, usize),
        /// Output (C, H, W): `oh = (ih-1)*s - pt - pb + kh (+ opad_h)`.
        oshape: (usize, usize, usize),
        /// (output_padding_h, output_padding_w).
        out_pad: (usize, usize),
    },
    /// Per-channel affine `y = scale[ch] * x + shift[ch]` over a (C, H, W)
    /// tensor — standalone BatchNormalization in inference form
    /// (#epoch-bab Phase D). Diagonal, so it composes with every tableau /
    /// backward path like a fixed exact gate with certified parameter error.
    ChannelAffine {
        /// Producer tensor id consumed.
        input: usize,
        /// Per-channel scale `w_bn / sqrt(var + eps)` (f64-folded).
        scale: Vec<f64>,
        /// Per-channel shift `b_bn - scale * mean`.
        shift: Vec<f64>,
        /// Certified RELATIVE error on `scale` vs the exact real fold.
        scale_rel_err: f64,
        /// Certified ABSOLUTE per-channel error on `shift`.
        shift_err: Vec<f64>,
        /// Tensor shape (C, H, W).
        shape: (usize, usize, usize),
    },
    /// Elementwise ReLU.
    Relu {
        /// Producer tensor id consumed.
        input: usize,
    },
    /// Elementwise Add of two tensors of identical flat size.
    Add {
        /// Left tensor id.
        lhs: usize,
        /// Right tensor id.
        rhs: usize,
    },
    /// Flatten (identity on the flat representation).
    Flatten {
        /// Producer tensor id consumed.
        input: usize,
    },
    /// Dense layer `y = W x + b`, `W` row-major (n_out, n_in).
    ///
    /// #cct-margin-row-admission: the HEAD Gemm may carry an inference-form
    /// `BatchNormalization` folded in by the builder (CNN7's
    /// `Gemm -> BatchNormalization -> Relu -> Gemm` head), exactly as `Conv`
    /// carries a folded BN. The fold is irrational (`gamma / sqrt(var + eps)`),
    /// so it cannot be exact in f64; its certified error travels with the op
    /// and the compiler charges it wherever the head weights enter a bound.
    /// An unfolded Gemm is EXACT: `bias_err` empty and `weight_rel_err` 0.0.
    /// The compiler refuses a fold error on the FINAL Gemm (every margin
    /// consumer reads it as exact).
    Gemm {
        /// Producer tensor id consumed.
        input: usize,
        /// Weights, row-major (n_out, n_in).
        weight: Vec<f64>,
        /// Bias (n_out).
        bias: Vec<f64>,
        /// Certified ABSOLUTE per-row error bound on `bias` vs the exact
        /// real-arithmetic fold. EMPTY means exact (no folded parameter);
        /// otherwise exactly one entry per output row.
        bias_err: Vec<f64>,
        /// Certified RELATIVE error bound on every `weight` entry vs the exact
        /// real fold, relative to the STORED weight (0.0 when nothing was
        /// folded).
        weight_rel_err: f64,
        /// (n_out, n_in).
        shape: (usize, usize),
    },
}

impl TwinOpSpec {
    /// Does this op carry a nonzero certified fold-error budget (a `Gemm`
    /// with a folded BatchNormalization)? `false` for every other op kind,
    /// whose error budgets the compiler handles in place.
    pub fn is_folded_gemm(&self) -> bool {
        match self {
            Self::Gemm {
                bias_err,
                weight_rel_err,
                ..
            } => *weight_rel_err != 0.0 || bias_err.iter().any(|e| *e != 0.0),
            _ => false,
        }
    }
}

/// #cct-margin-row-admission — ceiling on the ROOT FORWARD TABLEAU of a net
/// the lane will compile, in f64 ENTRIES of its widest `(tensor, n_in + 1)`
/// matrix.
///
/// The root forward (`root.rs`) materializes, per live tensor, a `(n, n_in+1)`
/// value matrix and a second deviation matrix of the same shape. Nothing else
/// bounds that allocation, and on the uncapped competition path ny has NO
/// memory governor (CLAUDE.md: `anon-rss 118 GB`, `global_oom`). The widest
/// tensor of every net the lane has banked evidence on sits under this cap:
///
/// | net                                   | widest x (n_in+1) |
/// |---------------------------------------|-------------------|
/// | CIFAR100_resnet_medium                | 4.4e7             |
/// | CIFAR100_resnet_large                 | 2.0e8             |
/// | TinyImageNet_resnet_medium            | 4.4e8 (41% of cap)|
/// | cct cifar10 cnn7 / wide cnn7          | 2.0e8 / 4.0e8     |
/// | relusplitter oval21 (all kw variants) | <= 2.6e7          |
/// | metaroom_2023 4cnn                    | 2.9e7             |
///
/// (TinyImageNet_resnet_medium: input 3x56x56 = 9408, widest tensor the
/// stride-2 stem conv's 64x27x27 = 46656, so 46656 x 9409 = 4.39e8, derived
/// from the ONNX graphs, not from an assumed 64x64 input.)
///
/// while the cct TinyImageNet CNN7 pair — structurally inside the family once
/// the FC-BN fold admits it — needs 3.2e9 / 6.4e9 entries (26 / 52 GB per f64
/// matrix, two matrices per live tensor): a guaranteed OOM on a 121 GiB host
/// and a lost verdict on any competition machine. `2^30` entries = 8 GiB per
/// f64 matrix. Refusing is fail-closed: the lane does not run and the internal
/// verifier keeps its whole budget.
pub const MAX_ROOT_TABLEAU_ENTRIES: usize = 1 << 30;

/// The full network description.
#[derive(Debug, Clone)]
pub struct TwinSpec {
    /// Flat input size (e.g. 3*32*32).
    pub n_in: usize,
    /// Ops in execution order. Exactly two `Gemm`s; the first is preceded by
    /// `Flatten` and followed by `Relu` then the final `Gemm` (validated by
    /// the compiler).
    pub ops: Vec<TwinOpSpec>,
}

impl TwinSpec {
    /// Checked flat geometry shared by source admission and the dense census.
    /// Structural/parameter validation remains the compiler's responsibility.
    pub fn root_tensor_sizes(&self) -> Option<Vec<usize>> {
        let mut sizes = Vec::with_capacity(self.ops.len().checked_add(1)?);
        sizes.push(self.n_in);
        for (k, op) in self.ops.iter().enumerate() {
            let input = |id: usize| {
                if id <= k {
                    sizes.get(id).copied()
                } else {
                    None
                }
            };
            let size = match op {
                TwinOpSpec::Conv {
                    input: id, oshape, ..
                }
                | TwinOpSpec::ConvTranspose {
                    input: id, oshape, ..
                } => {
                    input(*id)?;
                    oshape.0.checked_mul(oshape.1)?.checked_mul(oshape.2)?
                }
                TwinOpSpec::ChannelAffine {
                    input: id, shape, ..
                } => {
                    let size = shape.0.checked_mul(shape.1)?.checked_mul(shape.2)?;
                    if input(*id)? != size {
                        return None;
                    }
                    size
                }
                TwinOpSpec::Relu { input: id } | TwinOpSpec::Flatten { input: id } => input(*id)?,
                TwinOpSpec::Add { lhs, rhs } => {
                    let size = input(*lhs)?;
                    if input(*rhs)? != size {
                        return None;
                    }
                    size
                }
                TwinOpSpec::Gemm {
                    input: id, shape, ..
                } => {
                    if input(*id)? != shape.1 {
                        return None;
                    }
                    shape.0
                }
            };
            sizes.push(size);
        }
        Some(sizes)
    }
    /// Entries of the widest root-tableau matrix: `max tensor size x
    /// (n_in + 1)`, over tensor 0 (the input) and every op output, using the
    /// SAME size rule as the compiler's `tsize` walk (`net.rs`). `None` on a
    /// dangling tensor id or arithmetic overflow — the caller refuses.
    pub fn root_tableau_entries(&self) -> Option<usize> {
        let sizes = self.root_tensor_sizes()?;
        let widest = sizes.iter().copied().max()?;
        widest.checked_mul(self.n_in.checked_add(1)?)
    }

    /// Is the root tableau within [`MAX_ROOT_TABLEAU_ENTRIES`]? The single
    /// predicate shared by the compiler and the ny-cli admission probe, so the
    /// two cannot disagree about which nets are too wide to run.
    pub fn root_tableau_within_cap(&self) -> bool {
        self.root_tableau_entries()
            .is_some_and(|entries| entries <= MAX_ROOT_TABLEAU_ENTRIES)
    }
}
