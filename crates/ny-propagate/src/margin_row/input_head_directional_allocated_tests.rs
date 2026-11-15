// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored Source controls; exact arithmetic and functional bounds, not timing.
use super::*;

pub(super) const INPUTS: usize = 80;
pub(super) const HEADS: usize = 96;

pub(super) fn end() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

pub(super) fn budget() -> Budget {
    Budget {
        deadline: end(),
        cells: 0,
        masks: 0,
    }
}

pub(super) fn input(old: usize, permuted: bool) -> usize {
    if permuted {
        (11 * old + 9) % INPUTS
    } else {
        old
    }
}

pub(super) fn head(old: usize, permuted: bool) -> usize {
    if permuted {
        (17 * old + 13) % HEADS
    } else {
        old
    }
}

pub(super) fn spec(bias: f32, permuted: bool) -> InputHeadSpec {
    let epsilon = 1.0f32 / 4096.0;
    let delta = 1.0f32 / 256.0;
    let mut first = vec![0.0; INPUTS * HEADS];
    let mut final_weight = vec![0.0; HEADS * 2];
    for i in 0..INPUTS {
        for p in 0..INPUTS {
            first[input(i, permuted) * HEADS + head(p, permuted)] =
                epsilon + if i == p { 1.0 } else { 0.0 };
        }
        for block in 0..2 {
            for k in 0..8 {
                // Direct dyadic FLOAT32 constants, not a fitted/folded product.
                first[input(i, permuted) * HEADS + head(80 + 8 * block + k, permuted)] = epsilon
                    * (40.0 + delta)
                    + if i / 40 == block { 1.0 } else { 0.0 }
                    + if i == 40 * block + k { delta } else { 0.0 };
            }
        }
    }
    for p in 0..INPUTS {
        final_weight[2 * head(p, permuted)] = 8.0 + if p % 40 < 8 { delta } else { 0.0 };
    }
    for n in 80..HEADS {
        final_weight[2 * head(n, permuted)] = -1.0;
    }
    InputHeadSpec::new(
        INPUTS,
        HEADS,
        first,
        vec![0.0; HEADS],
        final_weight,
        vec![bias, 0.0],
        [131 + u8::from(permuted); 32],
    )
    .unwrap()
}

pub(super) fn source(bias: f32, permuted: bool) -> Source {
    Source::build(
        &spec(bias, permuted),
        &[-1.0; INPUTS],
        &[1.0; INPUTS],
        0,
        1,
        end(),
    )
    .unwrap()
}

pub(super) fn candidate(source: &Source, block: usize, permuted: bool) -> allocated::Candidate {
    let mut negative = (0..8)
        .map(|k| (head(80 + 8 * block + k, permuted), 1.0))
        .collect::<Vec<_>>();
    let mut positive = (40 * block..40 * (block + 1))
        .map(|p| {
            let h = head(p, permuted);
            // Strict slack for outward sums in the optional allocation validator.
            let mut mass = source.q[h].lo;
            for _ in 0..8 {
                mass = mass.next_down();
            }
            (h, mass)
        })
        .collect::<Vec<_>>();
    negative.sort_by_key(|&(h, _)| h);
    positive.sort_by_key(|&(h, _)| h);
    allocated::Candidate {
        negative: Direction { heads: negative },
        positive: signed::PositiveTerms(positive),
    }
}

pub(super) fn manual_cuts<'a>(
    source: &'a Source,
    permuted: bool,
    work: &mut Budget,
) -> Vec<CertifiedDirection<'a>> {
    (0..2)
        .map(|block| {
            let proposed = candidate(source, block, permuted);
            signed::certify_gray(source, proposed.negative, proposed.positive, vec![], work)
                .expect("each complete independently allocated row")
        })
        .collect()
}

pub(super) fn mixed_rows(request: &HeadFunctionalLpRequest) -> usize {
    request
        .rows
        .iter()
        .filter(|row| {
            row.coefficients
                .iter()
                .filter(|&&(j, a)| j >= INPUTS && a < 0.0)
                .count()
                >= 2
                && row
                    .coefficients
                    .iter()
                    .any(|&(j, a)| j >= INPUTS && a > 0.0)
        })
        .count()
}

pub(super) fn check_caps(report: &Refinement) {
    assert!(report.cuts <= MAX_CUTS * MAX_ROUNDS);
    assert!(report.attempted <= MAX_ATTEMPTS && report.masks <= MAX_MASKS);
    assert!(report.work_cells <= MAX_CELLS && report.rounds.len() <= MAX_ROUNDS);
    assert!(report.calls <= MAX_ROUNDS && report.native_replayed <= MAX_ROUNDS);
    assert!(report
        .rounds
        .iter()
        .all(|r| r.cuts <= MAX_CUTS && r.calls <= 1));
    assert_eq!(
        report.rounds.iter().map(|r| r.masks).sum::<usize>(),
        report.masks
    );
    assert_eq!(
        report.rounds.iter().map(|r| r.calls).sum::<usize>(),
        report.calls
    );
}

#[test]
fn allocated_dense_eighty_input_fixture_has_exact_float32_and_512_mask_reference() {
    let authored = spec(0.5, false);
    assert!(authored.first.iter().all(|&w| w > 0.0));
    let denominator = 1i64 << 23;
    let mut completed = 0usize;
    let mut combined_loss = 0i64;
    for block in 0..2 {
        let mut worst = 0i64;
        for mask in 0..256usize {
            let k = i64::from(mask.count_ones());
            let mut loss = 0i64;
            for i in 0..INPUTS {
                let inside = i / 40 == block;
                let special = inside && i % 40 < 8;
                let selected = special && mask & (1 << (i % 40)) != 0;
                let alpha_num = 2048 + if special { 1 } else { 0 };
                let positive_num = 81_928 + if inside { 4096 * alpha_num } else { 0 };
                let negative_num = k * 10_241
                    + if inside { k * (1 << 20) } else { 0 }
                    + if selected { 4096 } else { 0 };
                loss += (k * positive_num - 8 * negative_num).abs();
            }
            assert_eq!(loss, 8192 * k * (8 - k));
            worst = worst.max(loss);
            completed += 1;
        }
        assert_eq!(worst, denominator / 64);
        combined_loss += worst;
    }
    assert_eq!(completed, 512);
    assert_eq!(32 * (denominator / 2 - combined_loss), 15 * denominator);
    for i in 0..INPUTS {
        for g in 0..2 {
            for k in 0..8 {
                let expected = 10_241
                    + if i / 40 == g { 1 << 20 } else { 0 }
                    + if i == 40 * g + k { 4096 } else { 0 };
                assert_eq!(
                    f64::from(authored.first[i * HEADS + 80 + 8 * g + k]) * f64::from(1 << 20),
                    f64::from(expected)
                );
            }
        }
    }
    // Every product uses at most42 significant dyadic bits, so f64 is exact.
    for a in 0..HEADS {
        for b in a + 1..HEADS {
            assert!((1..INPUTS).any(|i| f64::from(authored.first[i * HEADS + a])
                * f64::from(authored.first[b])
                != f64::from(authored.first[i * HEADS + b]) * f64::from(authored.first[a])));
        }
    }
    // Subadditivity bounds each negative block by its allocated positive sum;
    // x=0 attains equality. The ideal minimum is the authored output bias.
    assert!(authored.first_bias.iter().all(|&v| v == 0.0));
    assert_eq!(authored.final_bias, vec![0.5, 0.0]);
}

#[test]
fn allocated_two_independent_rows_share_original_q_lp_and_close_actual_source() {
    for bias in [0.5, -0.5] {
        let s = source(bias, false);
        // Direct original FLOAT32 forward evaluation at the zero input:
        // every authored preactivation, ReLU and weighted term is exactly0.
        let hidden = s
            .spec
            .first_bias
            .iter()
            .map(|&b| b.max(0.0))
            .collect::<Vec<_>>();
        let mut zero_margin = s.spec.final_bias[0] - s.spec.final_bias[1];
        for (j, &r) in hidden.iter().enumerate() {
            zero_margin += (s.spec.final_weight[2 * j] - s.spec.final_weight[2 * j + 1]) * r;
        }
        assert_eq!(zero_margin, bias);
        let mut work = budget();
        let cuts = manual_cuts(&s, false, &mut work);
        assert_eq!(work.masks, 512);
        let model = triangle::Model::from_root_portfolio(&s, &mut None, &cuts, end()).unwrap();
        let request = model.request();
        assert!(request.validate());
        assert_eq!(request.feature_count, INPUTS);
        assert_eq!(request.bounds.len(), INPUTS + HEADS);
        assert_eq!(request.objective.len(), INPUTS + HEADS);
        assert!(request.objective[..INPUTS].iter().all(|&q| q == 0.0));
        for j in 0..HEADS {
            assert!(
                s.q[j].lo <= request.objective[INPUTS + j]
                    && request.objective[INPUTS + j] <= s.q[j].hi
            );
        }
        assert_eq!(mixed_rows(request), 2);
        assert_eq!(request.rows.len(), model.scalar_count() + 2);
        let hint = native::portfolio_hint(&model, &cuts, &[0.0; INPUTS], &mut work).unwrap();
        assert!(hint.row_duals.iter().rev().take(2).all(|&v| v > 0.99));
        let lower = model.replay(&s, &hint, 1.0, end()).unwrap();
        if bias > 0.0 {
            assert!(lower > 0.4 && lower <= 0.5, "{lower}");
        } else {
            assert!(lower <= -0.5, "original zero counterexample lost: {lower}");
        }
        assert!(work.cells <= MAX_CELLS);
    }
}

#[test]
fn allocated_generic_clustering_and_mass_assignment_survive_input_and_head_permutation() {
    for permuted in [false, true] {
        let s = source(0.5, permuted);
        let mut work = budget();
        let discovered = allocated::discover(&s, &mut work);
        assert!(discovered.len() >= 2);
        assert!(allocated::valid_allocation(&s, &discovered, &mut work));
        for g in 0..2 {
            let expected = (0..8)
                .map(|k| head(80 + 8 * g + k, permuted))
                .collect::<std::collections::BTreeSet<_>>();
            assert!(discovered.iter().any(|c| c
                .negative
                .heads
                .iter()
                .map(|&(h, _)| h)
                .collect::<std::collections::BTreeSet<_>>()
                == expected));
        }
        let cuts = discovered
            .into_iter()
            .take(MAX_CUTS)
            .map(|c| signed::certify_gray(&s, c.negative, c.positive, vec![], &mut work).unwrap())
            .collect::<Vec<_>>();
        assert!(cuts.len() >= 2);
        let model = triangle::Model::from_root_portfolio(&s, &mut None, &cuts, end()).unwrap();
        let hint = native::portfolio_hint(&model, &cuts, &[0.0; INPUTS], &mut work).unwrap();
        let lower = model.replay(&s, &hint, 1.0, end()).unwrap();
        assert!(lower > 0.4 && lower <= 0.5, "permuted{permuted}: {lower}");
        assert!(work.masks >= 512 && work.masks <= MAX_MASKS && work.cells <= MAX_CELLS);
    }
}

#[test]
fn allocated_automatic_source_portfolio_admits_a_completed_two_row_first_batch() {
    for permuted in [false, true] {
        let s = source(0.5, permuted);
        let (report, prefix) =
            refine_with_prefix(&s, s.baseline(end()).unwrap(), None, end(), end(), &|| None);
        assert!(
            report.lower > 0.4 && report.lower <= 0.5,
            "{}",
            report.lower
        );
        assert!(report.rounds.first().unwrap().cuts >= 2);
        assert!(report.cuts >= 2 && report.native_replayed >= 1 && prefix.is_some());
        assert_eq!(report.calls, 0);
        check_caps(&report);
    }
}
