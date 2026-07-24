// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// PropagateNetworkOptions now lives in ny-build (#1752).
pub use ny_build::PropagateNetworkOptions;

// GraphNetworkOptions + MissingOutputPolicy + CompoundNodePolicy now live in ny-build (#1752, #4173).
pub use ny_build::{CompoundNodePolicy, GraphNetworkOptions, MissingOutputPolicy};
