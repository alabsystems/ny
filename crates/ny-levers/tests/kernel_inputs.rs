// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use ny_levers::{decls::kernel_inputs, LeverDecl, LeverSet, RawLeverInputs};
use std::collections::BTreeMap;
use std::ffi::OsString;

fn inputs() -> [&'static LeverDecl; 3] {
    [
        &kernel_inputs::CONVTRANSPOSE_SOUND_F64_GPU,
        &kernel_inputs::CONV_SKIP_DEAD_F32,
        &kernel_inputs::FORWARD_LINEAR_MACS_PER_SEC,
    ]
}

#[test]
fn frozen_kernel_inputs_preserve_exact_os_bytes_and_lossy_receipt_view() {
    let mut values = vec![
        None,
        Some(OsString::new()),
        Some(OsString::from("0")),
        Some(OsString::from("1")),
        Some(OsString::from("true")),
        Some(OsString::from(" 1 ")),
        Some(OsString::from("18446744073709551616")),
        Some(OsString::from("340282366920938463463374607431768211455")),
        Some(OsString::from("é")),
    ];
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        values.push(Some(OsString::from_vec(vec![b'1', 255])));
    }
    for decl in inputs() {
        for value in &values {
            let captured = RawLeverInputs::capture_with(ny_levers::all(), |name| {
                if name == decl.name {
                    value.clone()
                } else {
                    None
                }
            });
            assert_eq!(captured.get(decl), value.as_deref());
            let resolved = LeverSet::resolve(ny_levers::all(), &captured);
            let observed = resolved.resolved(decl).expect("registered raw input");
            // Raw input retains bytes; its strict Unicode view matches the
            // legacy string readers. The JSON receipt separately renders Text
            // lossily and records whether the original input was Unicode.
            assert_eq!(
                captured.get(decl).and_then(|s| s.to_str()),
                value.as_deref().and_then(|s| s.to_str())
            );
            let receipt_text = value.as_ref().map(|s| s.to_string_lossy());
            assert_eq!(observed.value.as_str(), receipt_text.as_deref());
            assert_eq!(
                observed.env_utf8,
                value.as_deref().map(|s| s.to_str().is_some())
            );
        }
    }
}

#[test]
fn frozen_kernel_inputs_do_not_follow_later_input_changes() {
    let declarations = inputs();
    let mut ambient: BTreeMap<_, _> = declarations
        .iter()
        .enumerate()
        .map(|(i, decl)| (decl.name, OsString::from(i.to_string())))
        .collect();
    let captured =
        RawLeverInputs::capture_with(ny_levers::all(), |name| ambient.get(name).cloned());
    for decl in declarations {
        ambient.insert(decl.name, OsString::from("changed"));
    }
    for (i, decl) in declarations.iter().enumerate() {
        let original = OsString::from(i.to_string());
        assert_eq!(captured.get(decl), Some(original.as_os_str()));
        assert_ne!(
            captured.get(decl),
            ambient.get(decl.name).map(|s| s.as_os_str())
        );
    }
}

#[test]
fn inherited_kernel_inputs_cannot_become_automated_tuning_axes() {
    for decl in inputs() {
        assert!(ny_levers::all().get(decl.name).is_some());
        let sample = BTreeMap::from([(decl.name, String::from("1"))]);
        let refused = ny_levers::space::expand(&sample).expect_err("manual compatibility input");
        assert_eq!(refused.axis, decl.name);
        assert!(refused.because.contains("Class::Unsafe"));
    }
}
