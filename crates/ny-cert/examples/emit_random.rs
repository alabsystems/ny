// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Emit proof-carrying certificates for a range of deterministically-generated
//! ReLU-1 networks, for cross-repo verification by Clean's real external-cert
//! binary.
//!
//! Usage: `emit_random <seed_start> <count> <out_dir>`
//!
//! For each certifiable seed it writes `s<seed>_entailment.json` and
//! `s<seed>_farkas.json`, and prints one line per seed:
//! `seed=<n> bound=<m> file=<stem>` (or `seed=<n> SKIP overflow`). The exit code
//! is 0 unless an argument is malformed.

use ny_cert::generate::random_problem;
use ny_cert::{entailment_to_json, farkas_to_json, Rat};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: emit_random <seed_start> <count> <out_dir>");
        std::process::exit(2);
    }
    let seed_start: u64 = args[1].parse().expect("seed_start");
    let count: u64 = args[2].parse().expect("count");
    let out_dir = &args[3];
    std::fs::create_dir_all(out_dir).expect("create out_dir");

    let mut emitted = 0u64;
    let mut skipped = 0u64;
    for seed in seed_start..seed_start + count {
        let problem = random_problem(seed, 3, 4);
        // Certify at the network's own CROWN bound so the property is tight.
        let bound = match problem.certify(Rat::ZERO) {
            Ok(c) => c.lower_bound,
            Err(ny_cert::CrownError::ThresholdAboveBound { bound, .. }) => {
                match parse_bound(&bound) {
                    Some(b) => b,
                    None => {
                        println!("seed={seed} SKIP overflow");
                        skipped += 1;
                        continue;
                    }
                }
            }
            Err(_) => {
                println!("seed={seed} SKIP overflow");
                skipped += 1;
                continue;
            }
        };
        let certified = problem.certify(bound).expect("certify at own bound");
        let ent = match entailment_to_json(&certified.entailment) {
            Ok(j) => j,
            Err(_) => {
                println!("seed={seed} SKIP overflow");
                skipped += 1;
                continue;
            }
        };
        let far = farkas_to_json(&certified.farkas).expect("farkas json");
        let stem = format!("s{seed}");
        std::fs::write(
            format!("{out_dir}/{stem}_entailment.json"),
            serde_json::to_string_pretty(&ent).unwrap(),
        )
        .unwrap();
        std::fs::write(
            format!("{out_dir}/{stem}_farkas.json"),
            serde_json::to_string_pretty(&far).unwrap(),
        )
        .unwrap();
        println!(
            "seed={seed} bound={}/{} file={stem}",
            bound.num(),
            bound.den()
        );
        emitted += 1;
    }
    eprintln!("emit_random: {emitted} emitted, {skipped} skipped");
}

fn parse_bound(s: &str) -> Option<Rat> {
    let (n, d) = match s.split_once('/') {
        Some((n, d)) => (n.parse().ok()?, d.parse().ok()?),
        None => (s.parse().ok()?, 1),
    };
    Rat::new(n, d).ok()
}
