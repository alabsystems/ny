// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Re-exports from the shared test fixture module.

pub(super) use crate::test_fixtures::{
    load_avoice_contract, optional_test_model, require_test_model, require_test_model_with_hint,
    AVOICE_TEST_MODEL_HINT, TRANSFORMER_TEST_MODEL_HINT, WHISPER_TEST_MODEL_HINT,
};
