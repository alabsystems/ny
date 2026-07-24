// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Utility to export PyTorch Whisper to ONNX.
pub fn generate_whisper_export_script(model_size: &str) -> String {
    format!(
        r#"#!/usr/bin/env python3
"""Export Whisper model to ONNX for ny verification."""

import torch
import whisper

def export_whisper(model_size: str = "{model_size}", output_path: str = "whisper_{model_size}.onnx"):
    model = whisper.load_model(model_size)
    model.eval()
    mel = torch.randn(1, 80, 3000)
    torch.onnx.export(
        model.encoder,
        mel,
        output_path,
        input_names=["mel"],
        output_names=["encoder_output"],
        dynamic_axes={{"mel": {{0: "batch", 2: "time"}}}},
        opset_version=17,
    )
    print(f"Exported to {{output_path}}")

if __name__ == "__main__":
    export_whisper()
"#,
        model_size = model_size
    )
}
