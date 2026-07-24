// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use ny_onnx::{load_onnx_bytes_with_config, OnnxLoadConfig, ShapeInferencePolicy};

fuzz_target!(|data: &[u8]| {
    let config = OnnxLoadConfig::default().with_shape_inference_policy(ShapeInferencePolicy::Skip);
    let _ = load_onnx_bytes_with_config("fuzz.onnx", data, &config);
});
