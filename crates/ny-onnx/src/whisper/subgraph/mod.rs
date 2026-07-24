// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Subgraph extraction for compositional Whisper verification.
//!
//! Extracts attention and MLP subgraphs from encoder blocks for
//! independent bound propagation.

pub(crate) mod attention;
mod mlp;
