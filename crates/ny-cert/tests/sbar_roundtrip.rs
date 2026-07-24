// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Emit the SBAR (Pillar 2) attention support-bound certificate as Clean's
//! canonical entailment JSON, for the cross-repo round-trip against Clean's real
//! external-certificate verifier (`scripts/clean_sbar.sh`). Set
//! `NY_CERT_OUT_DIR` to dump the certificate.

use ny_cert::entailment_to_json;
use ny_cert::sbar::SimplexSupportLp;
use ny_cert::Rat;

fn r(n: i128, d: i128) -> Rat {
    Rat::new(n, d).unwrap()
}

#[test]
fn sbar_certificate_emits_clean_json() {
    // Section-1.2 example, upper bound: g=(v̄_1,v̄_2)=(1,11), p∈[1/10,9/10].
    let lp = SimplexSupportLp {
        g: vec![r(1, 1), r(11, 1)],
        p_lo: vec![r(1, 10), r(1, 10)],
        p_hi: vec![r(9, 10), r(9, 10)],
    };
    let cert = lp.certify_upper().unwrap();
    assert_eq!(cert.bound, r(10, 1));
    let json = entailment_to_json(&cert.entailment).unwrap();
    assert_eq!(json["type"], "entailment_certificate");
    assert_eq!(json["conclusion"]["kind"], "le");

    if let Ok(dir) = std::env::var("NY_CERT_OUT_DIR") {
        std::fs::write(
            format!("{dir}/ny_sbar_entailment.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
    }
}
