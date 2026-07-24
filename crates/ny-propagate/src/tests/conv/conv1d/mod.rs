// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conv1d propagation tests.

use super::*;
use ndarray::{arr1, ArrayD};

mod batched;
mod crown;
mod engine;
mod ibp;
mod network;
mod stride_padding;
mod transpose;
mod validation;
