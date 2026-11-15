// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Source-API strength control, not ONNX adapter, G3 or elapsed-time evidence.
use super::*;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

fn dense_spec(bias: f32, permuted: bool) -> InputHeadSpec {
    let (d, h) = (30, 78);
    let input = |i| if permuted { (11 * i + 9) % d } else { i };
    let head = |j| if permuted { (17 * j + 13) % h } else { j };
    let mut first = vec![0.0; d * h];
    let mut final_weight = vec![0.0; h * 2];
    for i in 0..d {
        for p in 0..d {
            first[input(i) * h + head(p)] = 1.0 / 64.0 + if i == p { 1.0 } else { 0.0 };
        }
        for block in 0..2 {
            for k in 0..24 {
                // Exact stored dyadics; this is not a folded/fitted product.
                let numerator = 983_105
                    + k
                    + if i / 15 == block { 1 << 22 } else { 0 }
                    + if i == block * 15 + k % 15 { 4096 } else { 0 }
                    + if i == block * 15 + (k + 1) % 15 {
                        64 * (k + 1)
                    } else {
                        0
                    };
                first[input(i) * h + head(30 + 24 * block + k)] = numerator as f32 / 4_194_304.0;
            }
        }
    }
    for p in 0..30 {
        let numerator: usize = 24 * 65_536
            + (0..24)
                .map(|k| {
                    (if k % 15 == p % 15 { 64 } else { 0 })
                        + if (k + 1) % 15 == p % 15 { k + 1 } else { 0 }
                })
                .sum::<usize>();
        final_weight[2 * head(p)] = numerator as f32 / 65_536.0;
    }
    for n in 30..78 {
        final_weight[2 * head(n)] = -1.0;
    }
    InputHeadSpec::new(
        d,
        h,
        first,
        vec![0.0; h],
        final_weight,
        vec![bias, 0.0],
        [145; 32],
    )
    .unwrap()
}

#[test]
fn allocation_projection_dense_thirty_input_six_rows_restore_source_strength() {
    for permuted in [false, true] {
        for bias in [0.5, -0.5] {
            let spec = dense_spec(bias, permuted);
            assert!(spec.first_bias.iter().all(|&v| v == 0.0));
            assert_eq!(spec.final_bias[0] - spec.final_bias[1], bias);
            let s = Source::build(&spec, &[-1.0; 30], &[1.0; 30], 0, 1, end()).unwrap();
            let mut work = Budget {
                deadline: end(),
                cells: 0,
                masks: 0,
            };
            let discovered = discover(&s, &mut work);
            assert_eq!(discovered.len(), 6);
            assert!(valid_allocation(&s, &discovered, &mut work));
            let cuts = discovered
                .into_iter()
                .map(|c| {
                    signed::certify_gray(&s, c.negative, c.positive, vec![], &mut work).unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(cuts.len(), 6);
            assert_eq!(work.masks, 1536);
            // This API admits at most MAX_CUTS per append, as production does.
            let mut prefix = None;
            let first =
                triangle::Model::from_root_portfolio(&s, &mut prefix, &cuts[..MAX_CUTS], end())
                    .unwrap();
            let retained_rows = first.request().rows.clone();
            let original_objective = first.request().objective.clone();
            assert_eq!(original_objective.len(), s.spec.inputs + s.spec.heads);
            assert!(original_objective[..s.spec.inputs]
                .iter()
                .all(|&q| q == 0.0));
            for (head, q) in s.q.iter().enumerate() {
                let expected = 0.5 * q.lo + 0.5 * q.hi;
                assert_eq!(
                    original_objective[s.spec.inputs + head].to_bits(),
                    expected.to_bits()
                );
            }
            prefix = Some(first.into_phase_prefix());
            let model =
                triangle::Model::from_root_portfolio(&s, &mut prefix, &cuts[MAX_CUTS..], end())
                    .unwrap();
            assert_eq!(model.request().objective, original_objective);
            assert_eq!(
                model.request().rows.len(),
                retained_rows.len() + cuts.len() - MAX_CUTS
            );
            for (old, retained) in retained_rows.iter().zip(&model.request().rows) {
                assert_eq!(old.lower.to_bits(), retained.lower.to_bits());
                assert_eq!(
                    old.coefficients
                        .iter()
                        .map(|&(j, a)| (j, a.to_bits()))
                        .collect::<Vec<_>>(),
                    retained
                        .coefficients
                        .iter()
                        .map(|&(j, a)| (j, a.to_bits()))
                        .collect::<Vec<_>>(),
                );
            }
            let mixed = model
                .request()
                .rows
                .iter()
                .filter(|row| {
                    row.coefficients
                        .iter()
                        .filter(|&&(j, a)| j >= s.spec.inputs && a < 0.0)
                        .count()
                        >= 2
                        && row
                            .coefficients
                            .iter()
                            .any(|&(j, a)| j >= s.spec.inputs && a > 0.0)
                })
                .count();
            assert_eq!(mixed, cuts.len());
            let hint = native::portfolio_hint(&model, &cuts, &[0.0; 30], &mut work).unwrap();
            let lower = model.replay(&s, &hint, 1.0, end()).unwrap();
            if bias > 0.0 {
                assert!(lower > 0.4 && lower <= 0.5, "permuted{permuted}: {lower}");
            } else {
                assert!(
                    lower <= -0.5,
                    "zero-input original counterexample lost: {lower}"
                );
            }
            assert!(work.cells <= MAX_CELLS && work.masks <= MAX_MASKS);
        }
    }
}
