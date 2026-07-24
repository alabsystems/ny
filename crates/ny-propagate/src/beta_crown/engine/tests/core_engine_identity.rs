// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;
use ny_core::{GemmEngine, NaiveCpuGemmEngine};

struct TaggedTestGemmEngine {
    tag: usize,
}

impl TaggedTestGemmEngine {
    fn new(tag: usize) -> Self {
        Self { tag }
    }
}

impl GemmEngine for TaggedTestGemmEngine {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        let _ = self.tag;
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }
}

// Use a non-ZST engine so pointer-identity assertions cannot collapse.
fn tagged_engine_arc_3089(tag: usize) -> Arc<dyn GemmEngine> {
    Arc::new(TaggedTestGemmEngine::new(tag))
}

#[test]
fn test_new_with_engine_preserves_engine_identity_3089() {
    let engine = tagged_engine_arc_3089(1);
    let verifier = BetaCrownVerifier::new_with_engine(BetaCrownConfig::default(), engine.clone());

    let stored_ref = verifier.engine().expect("stored engine ref");
    let stored_arc = verifier.engine_arc().expect("stored engine arc");

    assert!(std::ptr::eq(stored_ref, engine.as_ref()));
    assert!(Arc::ptr_eq(&stored_arc, &engine));
}

#[test]
fn test_with_config_from_preserves_engine_identity_3089() {
    let engine = tagged_engine_arc_3089(1);
    let original = BetaCrownVerifier::new_with_engine(BetaCrownConfig::default(), engine);
    let derived = original.with_config_from(BetaCrownConfig {
        timeout: Duration::from_secs(99),
        ..Default::default()
    });

    let derived_engine = derived.engine_arc().expect("engine must be inherited");
    let original_engine = original.engine_arc().expect("original engine must exist");

    assert!(Arc::ptr_eq(&derived_engine, &original_engine));
    assert_eq!(derived.config.timeout, Duration::from_secs(99));
}

#[test]
fn test_resolve_engine_prefers_call_site_engine_3089() {
    let stored = tagged_engine_arc_3089(1);
    let verifier = BetaCrownVerifier::new_with_engine(BetaCrownConfig::default(), stored.clone());

    let arg_engine = TaggedTestGemmEngine::new(2);
    let resolved = verifier
        .resolve_engine(Some(&arg_engine))
        .expect("argument engine should win");

    assert!(std::ptr::eq(resolved, &arg_engine as &dyn GemmEngine));
    assert!(!std::ptr::eq(resolved, stored.as_ref()));
}

#[test]
fn test_resolve_engine_falls_back_to_stored_engine_3089() {
    let stored = tagged_engine_arc_3089(1);
    let verifier = BetaCrownVerifier::new_with_engine(BetaCrownConfig::default(), stored.clone());

    let resolved = verifier
        .resolve_engine(None)
        .expect("stored engine should resolve");

    assert!(std::ptr::eq(resolved, stored.as_ref()));
}
