// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Emit SBAR (Pillar 2) attention support-bound certificates for a range of
//! random truncated-simplex LPs, for cross-repo verification by Clean's real
//! external-certificate binary.
//!
//! Usage: `emit_sbar <seed_start> <count> <out_dir>`

use ny_cert::entailment_to_json;
use ny_cert::generate::random_simplex_lp;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: emit_sbar <seed_start> <count> <out_dir>");
        std::process::exit(2);
    }
    let seed_start: u64 = args[1].parse().expect("seed_start");
    let count: u64 = args[2].parse().expect("count");
    let out_dir = &args[3];
    std::fs::create_dir_all(out_dir).expect("create out_dir");

    let mut emitted = 0u64;
    for seed in seed_start..seed_start + count {
        let lp = random_simplex_lp(seed, 6);
        let cert = match lp.certify_upper() {
            Ok(c) => c,
            Err(_) => continue,
        };
        let json = match entailment_to_json(&cert.entailment) {
            Ok(j) => j,
            Err(_) => continue,
        };
        std::fs::write(
            format!("{out_dir}/sbar_s{seed}.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
        emitted += 1;
    }
    eprintln!("emit_sbar: {emitted} emitted");
}
