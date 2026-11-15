// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
fn write_values(path: &Path, values: &[f64]) {
    std::fs::write(
        path,
        values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}
fn request_fixture(dir: &Path) -> serde_json::Value {
    let weights = dir.join("head.f64le");
    let bias = dir.join("bias.f64le");
    let gamma = dir.join("gamma.f64le");
    let beta = dir.join("beta.f64le");
    write_values(&weights, &[0.0, 0.0, 1.0, 1.0, -1.0, -1.0]);
    write_values(&bias, &[0.0, -0.25, -0.5]);
    write_values(&gamma, &[1.0, 1.0]);
    write_values(&beta, &[0.0, 0.0]);
    serde_json::json!({"vocab_size":3,"hidden_size":2,"weights_f64le":weights,"bias_f64le":bias,
        "arithmetic":"ExactReal","target":0,"lo":[-100.0,-3.0],"hi":[20.0,70.0],"anchor":[0.0,0.0],
        "final_normalization":{"kind":"LayerNorm","arithmetic":"ExactReal","input_domain":"pre_normalization_box",
        "gamma_f64le":gamma,"beta_f64le":beta,"epsilon":1e-5}})
}
fn write_request(path: &Path, value: &serde_json::Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
#[test]
fn head_cli_normalized_default_typed_arrival_exports_and_originally_replays_complete_cover() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("request.json");
    let evidence = dir.path().join("evidence.json");
    let request = request_fixture(dir.path());
    write_request(&path, &request);
    super::super::run(&path, None, Some(&evidence), 60, 1).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&evidence).unwrap()).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["normalization"], "LayerNorm");
    let rows = parsed["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["competitor"], 1);
    assert_eq!(rows[1]["competitor"], 2);
    assert!(rows.iter().all(|r| r["upper"].as_f64().unwrap() < 0.0));
    super::super::run(&path, Some(&evidence), None, 60, 1).unwrap();
    // Absence retains the affine post-normalization interpretation and schema2.
    let mut legacy = request;
    legacy
        .as_object_mut()
        .unwrap()
        .remove("final_normalization");
    write_request(&path, &legacy);
    let affine = dir.path().join("affine.json");
    super::super::run(&path, None, Some(&affine), 60, 1).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&affine).unwrap()).unwrap();
    assert_eq!(parsed["schema_version"], 2);
    assert!(parsed.get("rows").is_none());
}
#[test]
fn head_cli_normalized_original_parameter_poison_and_wrong_norm_type_refuse_replay() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("request.json");
    let evidence = dir.path().join("evidence.json");
    let mut request = request_fixture(dir.path());
    write_request(&path, &request);
    super::super::run(&path, None, Some(&evidence), 60, 1).unwrap();
    request["final_normalization"]["kind"] = serde_json::json!("RmsNorm");
    write_request(&path, &request);
    assert!(super::super::run(&path, Some(&evidence), None, 60, 1).is_err());
    let rms = dir.path().join("rms.json");
    super::super::run(&path, None, Some(&rms), 60, 1).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&rms).unwrap()).unwrap();
    assert!(parsed["rows"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["upper"].as_f64().unwrap() >= 0.0));
    request["final_normalization"]["kind"] = serde_json::json!("LayerNorm");
    write_request(&path, &request);
    write_values(&dir.path().join("gamma.f64le"), &[1.0, 2.0]);
    assert!(super::super::run(&path, Some(&evidence), None, 60, 1).is_err());
    write_values(&dir.path().join("gamma.f64le"), &[1.0, 1.0]);
    let mut forged: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&evidence).unwrap()).unwrap();
    forged["rows"][0]["upper"] = serde_json::json!(-1000.0);
    write_request(&evidence, &forged);
    assert!(super::super::run(&path, Some(&evidence), None, 60, 1).is_err());
}
#[test]
fn head_cli_normalized_nonreal_domain_and_invalid_epsilon_refuse_before_coefficient_access() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("request.json");
    let mut request = request_fixture(dir.path());
    request["weights_f64le"] = serde_json::json!(dir.path().join("absent.f64le"));
    request["arithmetic"] = serde_json::json!("SequentialBinary64");
    write_request(&path, &request);
    assert!(super::super::run(&path, None, None, 60, 1)
        .unwrap_err()
        .to_string()
        .contains("requires ExactReal"));
    request["arithmetic"] = serde_json::json!("ExactReal");
    request["final_normalization"]["epsilon"] = serde_json::json!(0.0);
    write_request(&path, &request);
    assert!(super::super::run(&path, None, None, 60, 1)
        .unwrap_err()
        .to_string()
        .contains("epsilon"));
    request["final_normalization"]["epsilon"] = serde_json::json!(1e-5);
    request["final_normalization"]["input_domain"] = serde_json::json!("post_normalization_box");
    write_request(&path, &request);
    assert!(super::super::run(&path, None, None, 60, 1).is_err());
    request["final_normalization"]["input_domain"] = serde_json::json!("pre_normalization_box");
    request["final_normalization"]["arithmetic"] = serde_json::json!("SequentialBinary64");
    write_request(&path, &request);
    assert!(super::super::run(&path, None, None, 60, 1).is_err());
}
#[test]
fn head_cli_normalized_low_original_bits_and_exact_ties_stay_open_after_replay() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("request.json");
    let mut request = request_fixture(dir.path());
    request["vocab_size"] = serde_json::json!(2);
    request["hidden_size"] = serde_json::json!(1);
    request["lo"] = serde_json::json!([1.0]);
    request["hi"] = serde_json::json!([1.0]);
    request["anchor"] = serde_json::json!([1.0]);
    request["final_normalization"]["kind"] = serde_json::json!("RmsNorm");
    write_values(&dir.path().join("head.f64le"), &[1.0, 1.0 + 2f64.powi(-40)]);
    write_values(&dir.path().join("bias.f64le"), &[0.0, 0.0]);
    write_values(&dir.path().join("gamma.f64le"), &[1.0]);
    write_values(&dir.path().join("beta.f64le"), &[0.0]);
    write_request(&path, &request);
    let evidence = dir.path().join("lowbits.json");
    super::super::run(&path, None, Some(&evidence), 60, 1).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&evidence).unwrap()).unwrap();
    assert!(parsed["rows"][0]["upper"].as_f64().unwrap() > 0.0);
    write_values(&dir.path().join("head.f64le"), &[1.0, 1.0]);
    let tied = dir.path().join("tied.json");
    super::super::run(&path, None, Some(&tied), 60, 1).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(&tied).unwrap()).unwrap();
    assert_eq!(parsed["rows"][0]["upper"].as_f64().unwrap(), 0.0);
}
