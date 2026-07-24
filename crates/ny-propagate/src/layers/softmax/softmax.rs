// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Softmax layer shim module.
//!
//! The implementation lives in `layer.rs`; this module keeps a stable
//! `softmax` submodule path for builds that expect it after refactors.

pub use super::layer::SoftmaxLayer;

#[cfg(test)]
mod tests {
    use super::SoftmaxLayer;
    use crate::layers::softmax::{CausalSoftmaxLayer, LogSoftmaxLayer, LogSumExpLayer};

    #[test]
    fn softmax_shim_exports_compile() {
        let _ = SoftmaxLayer::new(-1);
        let _ = LogSoftmaxLayer::new(-1);
        let _ = LogSumExpLayer::new(vec![-1], false);
        let _ = CausalSoftmaxLayer::new(-1);
    }
}
