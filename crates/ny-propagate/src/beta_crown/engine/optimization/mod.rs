// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Joint α/β/λ optimization and bound computation.

mod bound_compute;
mod gradients;
mod intermediate_merge;
mod optimize_loop;
mod validate;

#[cfg(test)]
mod optimize_loop_regressions;
#[cfg(test)]
mod optimize_loop_tests;
