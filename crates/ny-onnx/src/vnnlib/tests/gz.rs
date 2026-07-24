// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::load_vnnlib;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;
use tempfile::tempdir;

#[ntest::timeout(10000)]
#[test]
fn load_vnnlib_supports_gzip() {
    let dir = tempdir().unwrap();
    let vnnlib_gz_path = dir.path().join("prop.vnnlib.gz");

    let vnnlib_content = r#"
; Property with label: 0.
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))
(assert (<= Y_0 0.0))
"#;

    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(vnnlib_content.as_bytes()).unwrap();
    let compressed = enc.finish().unwrap();
    std::fs::write(&vnnlib_gz_path, compressed).unwrap();

    let spec = load_vnnlib(&vnnlib_gz_path).unwrap();
    assert_eq!(spec.num_inputs, 1);
    assert_eq!(spec.num_outputs, 1);
    assert_eq!(spec.output_constraints.len(), 1);
}
