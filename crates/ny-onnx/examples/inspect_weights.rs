// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Example: Inspect weights in a PyTorch model file.
//!
//! Usage: cargo run --example inspect_weights --features pytorch -- <model_path>

use std::env;

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_path>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Inspecting weights from: {}", model_path);

    #[cfg(feature = "pytorch")]
    {
        use ny_onnx::pytorch::load_pytorch;

        match load_pytorch(model_path) {
            Ok(weights) => {
                println!(
                    "
=== Weights Summary ==="
                );
                println!("Total tensors: {}", weights.len());

                // Group weights by prefix
                let mut prefixes: std::collections::HashMap<String, Vec<String>> =
                    std::collections::HashMap::new();

                for name in weights.keys() {
                    let prefix = name.split('.').next().unwrap_or("").to_string();
                    prefixes.entry(prefix).or_default().push(name.to_string());
                }

                println!(
                    "
=== Weight Groups ==="
                );
                let mut prefix_list: Vec<_> = prefixes.iter().collect();
                prefix_list.sort_by_key(|(k, _)| *k);
                for (prefix, names) in prefix_list {
                    println!(
                        "
[{}] ({} tensors)",
                        prefix,
                        names.len()
                    );
                    for name in names.iter().take(5) {
                        if let Some(w) = weights.get(name) {
                            println!("  {} - shape {:?}", name, w.shape());
                        }
                    }
                    if names.len() > 5 {
                        println!("  ... and {} more", names.len() - 5);
                    }
                }

                // Show total parameters
                let total_params: usize = weights.iter().map(|(_, w)| w.len()).sum();
                println!(
                    "
=== Total Parameters ==="
                );
                println!("{} ({:.1}M)", total_params, total_params as f64 / 1e6);
            }
            Err(e) => {
                eprintln!("Error loading weights: {}", e);
                std::process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "pytorch"))]
    {
        eprintln!("PyTorch support not enabled. Rebuild with --features pytorch");
        std::process::exit(1);
    }
}
