// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-env-changed=CC");

    if env::var_os("CARGO_FEATURE_ORT").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
        || env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("gnu")
        || env::var("HOST") != env::var("TARGET")
    {
        return;
    }

    // ort-sys links static ONNX Runtime with `-lstdc++`. Some compiler shims
    // accept that flag while selecting a different C++ ABI, and runtime-only
    // installations may lack the development `libstdc++.so` symlink. Link the
    // discovered GNU runtime under an unambiguous private name.
    let compiler = env::var_os("CC").unwrap_or_else(|| "cc".into());
    if let Some(runtime) = compiler_file(&compiler, "libstdc++.so.6")
        .filter(|path| path.is_absolute() && path.exists())
    {
        expose_gnu_runtime(&runtime);
    }
}

fn expose_gnu_runtime(runtime: &Path) {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR"));
    // Use a private link name so compiler drivers cannot special-case
    // `-lstdc++` and silently select a different C++ standard library.
    let shim = out_dir.join("libny_stdcxx_runtime.so");
    if shim.symlink_metadata().is_ok() {
        fs::remove_file(&shim).expect("remove stale libstdc++ linker shim");
    }
    fs::copy(runtime, &shim).expect("create libstdc++ linker shim");

    // Search paths and native libraries from library build scripts propagate
    // to final dependents, so ny-cli also receives the GNU runtime under this
    // private name. The final absolute argument covers ny-onnx's own unit-test
    // target, where this package's native libraries precede dependencies under
    // `--as-needed`.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=ny_stdcxx_runtime");
    println!("cargo:rustc-link-arg={}", runtime.display());
}

fn compiler_file(compiler: &std::ffi::OsStr, name: &str) -> Option<PathBuf> {
    let output = Command::new(compiler)
        .arg(format!("-print-file-name={name}"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    (!path.is_empty() && path != name).then(|| PathBuf::from(path))
}
