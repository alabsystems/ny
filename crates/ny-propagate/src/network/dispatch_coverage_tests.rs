// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Backward CROWN dispatch coverage matrix test (#3424).
//!
//! Validates that each dispatch site handles exactly the expected set of
//! Layer variants. When the canonical dispatch (`dispatch_backward_layer`)
//! is exhaustive, this test catches drift: adding or removing a Layer
//! variant from any site's match block causes a test failure.
//!
//! Each `MATCH_BASED_SITES` entry specifies `expected_explicit` — the Layer
//! variants explicitly referenced in the site's dispatch match. Layers not
//! in `expected_explicit` are handled by the site's catch-all arm.
//!
//! `DELEGATING_SITES` entries verify that the site calls `dispatch_backward_layer`
//! and only overrides the expected site-specific layers.
//!
//! `SITE_SPECIFIC_ONLY_SITES` cover refactors where the coordinator keeps the
//! site-specific `Layer::*` handling but the actual `dispatch_backward_layer`
//! call moved into a shared helper.
//!
//! Source parsing utilities: [`super::dispatch_coverage_parser`].
//! Site data: [`super::dispatch_coverage_data`].

use std::collections::BTreeSet;

use super::dispatch_coverage_data::{
    DelegatingSiteExpectation, DispatchSiteExpectation, SiteSpecificOnlyExpectation,
    CANONICAL_SITE, DELEGATING_SITES, MATCH_BASED_SITES, SITE_SPECIFIC_ONLY_SITES,
};
use super::dispatch_coverage_parser::{
    extract_dispatch_signature, extract_layer_references, find_function_impl_start,
    find_matching_brace,
};

fn validate_match_site(
    site: &DispatchSiteExpectation,
    canonical: &BTreeSet<String>,
) -> Option<String> {
    let signature =
        extract_dispatch_signature(site.source.as_str(), site.fn_marker, site.match_index);
    let expected = if site.exhaustive {
        canonical.clone()
    } else {
        site.expected_explicit
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    };

    if signature.explicit_layers == expected {
        return None;
    }

    let extra: Vec<_> = signature
        .explicit_layers
        .difference(&expected)
        .cloned()
        .collect();
    let missing: Vec<_> = expected
        .difference(&signature.explicit_layers)
        .cloned()
        .collect();
    let label = if site.exhaustive {
        "expected (canonical)"
    } else {
        "expected explicit"
    };
    Some(format!(
        "site '{}'\n\
         {}: {:?}\n\
         actual explicit:   {:?}\n\
         added: {:?}\n\
         removed: {:?}",
        site.name, label, expected, signature.explicit_layers, extra, missing,
    ))
}

fn validate_delegating_site(site: &DelegatingSiteExpectation) -> Option<String> {
    let source = site.source.as_str();
    let fn_start = find_function_impl_start(source, site.fn_marker)
        .unwrap_or_else(|| panic!("function marker not found: {}", site.fn_marker));
    let fn_open_brace = source[fn_start..]
        .find('{')
        .map(|offset| fn_start + offset)
        .unwrap_or_else(|| panic!("opening brace not found for: {}", site.fn_marker));
    let fn_close_brace = find_matching_brace(source, fn_open_brace)
        .unwrap_or_else(|| panic!("closing brace not found for: {}", site.fn_marker));
    let fn_body = &source[(fn_open_brace + 1)..fn_close_brace];

    if !fn_body.contains("dispatch_backward_layer") {
        return Some(format!(
            "delegating site '{}' does not call dispatch_backward_layer",
            site.name
        ));
    }

    let layer_refs = extract_layer_references(fn_body);
    let expected: BTreeSet<String> = site
        .expected_site_specific
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    if layer_refs == expected {
        return None;
    }

    let extra: Vec<_> = layer_refs.difference(&expected).cloned().collect();
    let missing: Vec<_> = expected.difference(&layer_refs).cloned().collect();
    Some(format!(
        "delegating site '{}'\n\
         expected site-specific: {:?}\n\
         actual Layer:: refs: {:?}\n\
         extra: {:?}\nmissing: {:?}",
        site.name, expected, layer_refs, extra, missing,
    ))
}

fn validate_site_specific_only_site(site: &SiteSpecificOnlyExpectation) -> Option<String> {
    let source = site.source.as_str();
    let fn_start = find_function_impl_start(source, site.fn_marker)
        .unwrap_or_else(|| panic!("function marker not found: {}", site.fn_marker));
    let fn_open_brace = source[fn_start..]
        .find('{')
        .map(|offset| fn_start + offset)
        .unwrap_or_else(|| panic!("opening brace not found for: {}", site.fn_marker));
    let fn_close_brace = find_matching_brace(source, fn_open_brace)
        .unwrap_or_else(|| panic!("closing brace not found for: {}", site.fn_marker));
    let fn_body = &source[(fn_open_brace + 1)..fn_close_brace];

    let layer_refs = extract_layer_references(fn_body);
    let expected: BTreeSet<String> = site
        .expected_layer_refs
        .as_slice()
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    if layer_refs == expected {
        return None;
    }

    let extra: Vec<_> = layer_refs.difference(&expected).cloned().collect();
    let missing: Vec<_> = expected.difference(&layer_refs).cloned().collect();
    Some(format!(
        "site-specific-only '{}'\n\
         expected Layer:: refs: {:?}\n\
         actual Layer:: refs: {:?}\n\
         extra: {:?}\nmissing: {:?}",
        site.name, expected, layer_refs, extra, missing,
    ))
}

#[test]
fn backward_dispatch_coverage_matrix_matches_step_a_baseline() {
    let canonical = extract_dispatch_signature(
        CANONICAL_SITE.source.as_str(),
        CANONICAL_SITE.fn_marker,
        CANONICAL_SITE.match_index,
    );
    assert!(
        !canonical.explicit_layers.is_empty(),
        "canonical dispatch has no Layer variants — parser broken?"
    );

    let canonical_set = &canonical.explicit_layers;
    let mut mismatches: Vec<String> = MATCH_BASED_SITES
        .iter()
        .filter_map(|site| validate_match_site(site, canonical_set))
        .collect();
    mismatches.extend(DELEGATING_SITES.iter().filter_map(validate_delegating_site));
    mismatches.extend(
        SITE_SPECIFIC_ONLY_SITES
            .iter()
            .filter_map(validate_site_specific_only_site),
    );

    assert!(
        mismatches.is_empty(),
        "dispatch coverage drift detected:\n\n{}",
        mismatches.join("\n\n")
    );
}
