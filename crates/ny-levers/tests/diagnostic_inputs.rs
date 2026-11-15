// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use ny_levers::{decls, LeverSet, RawLeverInputs};
use std::ffi::OsString;

#[test]
fn diagnostic_paths_preserve_absence_empty_and_unicode_in_the_frozen_input() {
    for decl in [
        &decls::diagnostics::HULL_RESOURCE_ORIGINAL_REQUEST,
        &decls::diagnostics::GROUPED_ORIGINAL_PROBE_REQUEST,
        &decls::onnx::VIT_BENCH,
    ] {
        for value in [
            None,
            Some(OsString::new()),
            Some(OsString::from("fixture/é.json")),
        ] {
            let captured = RawLeverInputs::capture_with(ny_levers::all(), |name| {
                (name == decl.name).then(|| value.clone()).flatten()
            });
            assert_eq!(captured.get(decl), value.as_deref());
            let frozen = LeverSet::resolve(ny_levers::all(), &captured);
            assert_eq!(
                frozen.resolved(decl).unwrap().value.as_str(),
                value.as_deref().and_then(std::ffi::OsStr::to_str)
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn diagnostic_os_paths_survive_non_unicode_while_the_vit_string_view_declines_it() {
    use std::os::unix::ffi::OsStringExt;
    for decl in [
        &decls::diagnostics::HULL_RESOURCE_ORIGINAL_REQUEST,
        &decls::diagnostics::GROUPED_ORIGINAL_PROBE_REQUEST,
        &decls::onnx::VIT_BENCH,
    ] {
        let value = OsString::from_vec(vec![b'/', b't', 255]);
        let captured = RawLeverInputs::capture_with(ny_levers::all(), |name| {
            (name == decl.name).then(|| value.clone())
        });
        assert_eq!(captured.get(decl), Some(value.as_os_str()));
        assert!(captured
            .get(decl)
            .and_then(std::ffi::OsStr::to_str)
            .is_none());
        let frozen = LeverSet::resolve(ny_levers::all(), &captured);
        assert_eq!(frozen.resolved(decl).unwrap().env_utf8, Some(false));
    }
}
