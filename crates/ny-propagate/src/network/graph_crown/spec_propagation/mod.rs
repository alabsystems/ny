// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Spec-guided CROWN backward propagation for graph networks.
//!
//! This module provides the builder API (`SpecCrownRequest`) and the core
//! backward coordinator loop. The combinatorial wrapper functions that
//! previously lived here were replaced by the builder in #4220.

mod core;
mod fallback;
mod patches;
pub(crate) mod request;
mod setup;

pub(crate) use self::request::SpecCrownRequest;
pub(crate) use self::setup::collect_intermediate_bounds;
