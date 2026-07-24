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
    Gemm {
        /// Producer tensor id consumed.
        input: usize,
        /// Weights, row-major (n_out, n_in).
        weight: Vec<f64>,
        /// Bias (n_out).
        bias: Vec<f64>,
        /// (n_out, n_in).
        shape: (usize, usize),
    },
}

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
