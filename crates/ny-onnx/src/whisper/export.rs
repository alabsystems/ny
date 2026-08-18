// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Generate a script that exports a PyTorch Whisper encoder to ONNX.
///
/// `model_size` is serialized as a Python-compatible JSON string literal, so
/// arbitrary library callers cannot inject Python source through this value.
pub fn generate_whisper_export_script(model_size: &str) -> String {
    let model_size_literal =
        serde_json::to_string(model_size).expect("serializing a Rust string cannot fail");
    format!(
        r#"#!/usr/bin/env python3
"""Export a Whisper encoder to ONNX for ny loading and block analysis."""

import torch
import whisper
import onnx
from typing import Optional

def export_whisper(model_size: str = {model_size_literal}, output_path: Optional[str] = None):
    if output_path is None:
        output_path = f"whisper_{{model_size}}.onnx"
    model = whisper.load_model(model_size, device="cpu")
    model.eval()
    # Whisper positional embeddings fix the post-convolution audio context.
    # Derive both dimensions so variants such as large-v3 (128 mel bins) work.
    mel = torch.randn(1, model.dims.n_mels, model.dims.n_audio_ctx * 2)
    torch.onnx.export(
        model.encoder,
        mel,
        output_path,
        input_names=["mel"],
        output_names=["encoder_output"],
        # The modern dynamo exporter prefers torch.export dynamic-shape
        # constraints. A positional tuple avoids coupling this specification
        # to Whisper's internal Python parameter name.
        dynamic_shapes=({{0: torch.export.Dim("batch")}},),
        # Torch 2.6-2.8 did not yet default to the dynamo exporter, and the
        # legacy exporter rejects dynamic_shapes.
        dynamo=True,
        # Keep large encoders below protobuf's 2 GiB message limit. The ONNX
        # file and its generated data sidecar must remain together.
        external_data=True,
        opset_version=18,
    )
    onnx.checker.check_model(output_path, full_check=True)
    print(f"Exported to {{output_path}}")

if __name__ == "__main__":
    export_whisper()
"#,
        model_size_literal = model_size_literal
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_size_is_serialized_as_data_not_python_source() {
        let payload = "tiny\"\nraise SystemExit()";
        let literal = serde_json::to_string(payload).expect("serialize test payload");
        let script = generate_whisper_export_script(payload);
        assert!(script.contains(&format!("model_size: str = {literal}")));
        assert!(!script.lines().any(|line| line == "raise SystemExit()"));
        assert!(script.contains("device=\"cpu\""));
        assert!(script.contains("model.dims.n_mels"));
        assert!(script.contains("model.dims.n_audio_ctx * 2"));
        assert!(script.contains("dynamic_shapes=({0: torch.export.Dim(\"batch\")},)"));
        assert!(!script.contains("dynamic_axes"));
        assert!(script.contains("dynamo=True"));
        assert!(script.contains("external_data=True"));
        assert!(script.contains("opset_version=18"));
        assert!(script.contains("onnx.checker.check_model(output_path, full_check=True)"));
    }
}
