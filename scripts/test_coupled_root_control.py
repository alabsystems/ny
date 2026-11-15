#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Exact rational control for intersecting two certified root directions.

Standard library only. This proves the specified ideal continuous-box control
by a finite max-affine identity and matching witnesses; it runs no LP solver,
NY, AY, compiler, numerical checker, GPU, or benchmark.
"""
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
import unittest

SIZE = 7
SAFE_BIAS = F(25, 2)
UNSAFE_BIAS = F(23, 2)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def heads(x):
    require(len(x) >= 3, 'cycle must have at least three vertices')
    return tuple(max(F(0), value - x[(j + 1) % len(x)]) for j, value in enumerate(x))


def mask_coefficients(mask, weights):
    """Coefficients of sum_j s_j*w_j*(x_j-x_next) in original input order."""
    n = len(weights)
    require(n >= 3 and 0 <= mask < (1 << n) and all(w >= 0 for w in weights), 'invalid nonnegative direction/mask')
    active = tuple(weight if mask & (1 << j) else F(0) for j, weight in enumerate(weights))
    return tuple(active[j] - active[(j - 1) % n] for j in range(n))


def all_mask_supports(weights):
    # sum w*ReLU(t) = max_s sum s*w*t for NONNEGATIVE w. Each affine form's
    # support on [-1,1]^n is exactly sum |coefficient|, including every input.
    return tuple(sum(map(abs, mask_coefficients(mask, weights)), F(0)) for mask in range(1 << len(weights)))


def complete_cut(weights, records):
    """Replay complete numeric masks, not a prefix maximum or claimed endpoint."""
    count = 1 << len(weights)
    require(len(records) == count and {mask for mask, _ in records} == set(range(count)), 'missing or duplicate mask')
    for mask, upper in records:
        require(upper == sum(map(abs, mask_coefficients(mask, weights)), F(0)), 'wrong affine support')
    return -max(upper for _, upper in records)


def vertices(n):
    return itertools.product((F(-1), F(1)), repeat=n)


def triangle_valid(x, r, mask=None):
    if len(x) != len(r) or any(not F(-1) <= value <= F(1) for value in x):
        return False
    for j, value in enumerate(r):
        t = x[j] - x[(j + 1) % len(x)]
        if not (0 <= value <= 2 and value >= t and 2 * value <= 2 + t):
            return False
        if mask is not None and ((mask & (1 << j) and not (t >= 0 and value == t))
                                 or (not mask & (1 << j) and not (t <= 0 and value == 0))):
            return False
    return True


def proper_subset_mixture(mask, n=SIZE):
    """An exact two-point graph-hull witness for every proper head subset."""
    require(0 <= mask < (1 << n) - 1, 'full odd cycle is not a forest')
    neighbors = [[] for _ in range(n)]
    for j in range(n):
        if mask & (1 << j):
            k = (j + 1) % n
            neighbors[j].append(k)
            neighbors[k].append(j)
    colors = {}
    for start in range(n):
        if start in colors:
            continue
        colors[start], todo = F(1), [start]
        while todo:
            j = todo.pop()
            for k in neighbors[j]:
                if k in colors:
                    require(colors[k] == -colors[j], 'not a bipartite selected forest')
                else:
                    colors[k] = -colors[j]
                    todo.append(k)
    x = tuple(colors[j] for j in range(n))
    opposite = tuple(-value for value in x)
    r, opposite_r = heads(x), heads(opposite)
    require(all((a + b) / 2 == 0 for a, b in zip(x, opposite)), 'incorrect mean input')
    require(all((r[j] + opposite_r[j]) / 2 == 1 for j in range(n) if mask & (1 << j)), 'incorrect selected head mean')
    return x, opposite


def cancellation_components(rows, minimum=7, maximum=8):
    """Reference SEARCH candidates, not production NY or certificate authority.

    Components of opposite-sign first-layer interaction find both independent
    supports without adjacent head indices or positive prefix point-violation.
    Large/dense components need a separately bounded beam/growth strategy.
    """
    require(rows and len({len(row) for row in rows}) == 1, 'invalid row matrix')
    neighbors = [set() for _ in rows]
    for j, row in enumerate(rows):
        for k in range(j):
            if sum((a * b for a, b in zip(row, rows[k])), F(0)) < 0:
                neighbors[j].add(k)
                neighbors[k].add(j)
    unseen, result = set(range(len(rows))), []
    while unseen:
        start = min(unseen)
        component, todo = set(), [start]
        while todo:
            j = todo.pop()
            if j in component:
                continue
            component.add(j)
            todo.extend(neighbors[j] - component)
        unseen -= component
        if minimum <= len(component) <= maximum:
            result.append(tuple(sorted(component)))
    return tuple(result)


def two_cycle_rows():
    rows = []
    for block in range(2):
        for j in range(SIZE):
            row = [F(0)] * (2 * SIZE)
            row[block * SIZE + j] = F(1)
            row[block * SIZE + (j + 1) % SIZE] = F(-1)
            rows.append(tuple(row))
    return tuple(rows)


def control():
    weights = (F(1),) * SIZE
    supports = all_mask_supports(weights)
    cut = complete_cut(weights, list(enumerate(supports)))
    endpoints = [(sum(heads(x), F(0)), x) for x in vertices(SIZE)]
    endpoint_upper, endpoint = max(endpoints)
    require(-cut == endpoint_upper == 6, 'continuous support and endpoint disagree')
    outer_x, outer_r = (F(0),) * SIZE, (F(1),) * SIZE
    endpoint_r = heads(endpoint)
    endpoint_mask = sum(1 << j for j in range(SIZE) if endpoint[j] > endpoint[(j + 1) % SIZE])
    require(triangle_valid(endpoint, endpoint_r, endpoint_mask) and triangle_valid(outer_x, outer_r), 'bad attaining witness')
    require(sum(outer_r) == SIZE and heads(outer_x) != outer_r, 'outer witness mistaken for a network witness')
    # Every split-cycle mask has its own weak-dual floor. The unselected other
    # cycle has triangle upper seven. A matching feasible outer witness makes
    # these specified whole covers' scalar floor exact, not merely loose.
    scalar_a = min(SAFE_BIAS - upper - SIZE for upper in supports)
    scalar_b = scalar_a
    witness_scalar = SAFE_BIAS - sum(endpoint_r) - sum(outer_r)
    joint = SAFE_BIAS + cut + cut
    unsafe = UNSAFE_BIAS - 2 * sum(endpoint_r)
    require(scalar_a == scalar_b == witness_scalar == F(-1, 2), 'wrong separate cover floor')
    require(joint == SAFE_BIAS - 2 * endpoint_upper == F(1, 2), 'joint cut proof not tight')
    require(unsafe == F(-1, 2), 'unsafe sibling was lost')
    for mask in range((1 << SIZE) - 1):
        proper_subset_mixture(mask)
    components = cancellation_components(two_cycle_rows())
    require(components == (tuple(range(SIZE)), tuple(range(SIZE, 2 * SIZE))), 'reference search missed one support')
    return dict(
        schema='ny_coupled_root_exact_control_v1', arithmetic='fractions.Fraction; exact dyadics',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='specified two disjoint7-cycle ideal ReLU control and A/B covers; not all possible mixed covers',
        continuous_box_proof='sum positive ReLU terms=max of all affine masks; affine box support=sum absolute coefficients',
        cycle=dict(inputs=SIZE, masks=len(supports), endpoint_assignments=len(endpoints),
                   affine_mask_upper=[str(value) for value in supports], exact_upper=str(endpoint_upper),
                   proper_subset_hull_mixtures_checked=(1 << SIZE) - 1),
        two_cycles=dict(effective_inputs=2 * SIZE, heads=2 * SIZE, safe_bias=str(SAFE_BIAS),
            triangle_floor=str(SAFE_BIAS - 2 * SIZE), cover_a_floor=str(scalar_a), cover_b_floor=str(scalar_b),
            maximum_separate_scalar_floor=str(max(scalar_a, scalar_b)),
            geometry_cut_a_lower=str(cut), geometry_cut_b_lower=str(cut),
            joint_cut_floor=str(joint), exact_network_minimum=str(joint),
            split_attainer=dict(x=[str(v) for v in endpoint], r=[str(v) for v in endpoint_r], mask=endpoint_mask),
            unsplit_outer_attainer=dict(x=[str(v) for v in outer_x], r=[str(v) for v in outer_r])),
        unsafe_sibling=dict(bias=str(UNSAFE_BIAS), exact_network_minimum=str(unsafe),
            actual_input=[str(v) for v in endpoint + endpoint], actual_heads=[str(v) for v in endpoint_r + endpoint_r],
            attained_margin=str(unsafe)),
        reference_search=dict(method='negative inner-product connected components, completed7..8-head supports',
                              candidates=[list(group) for group in components], certificate_authority=False),
        production_source_executed=False, source_fp_error_qualified=False, formal_proof_claim=False,
        solver_executions=0, checker_executions=0, source_builds=0, benchmark_or_score_claim=False)


def verify_report(path):
    require(not Path(path).is_symlink(), 'report symlink refused')
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON: ' + value)
    actual = json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=nonfinite)
    require(actual == control(), 'report differs from freshly checked exact control')


class ExactControl(unittest.TestCase):
    def test_all_continuous_masks_have_exact_box_support_and_attainers(self):
        weights = (F(1),) * SIZE
        for mask in range(1 << SIZE):
            coefficients = mask_coefficients(mask, weights)
            x = tuple(F(1) if coefficient >= 0 else F(-1) for coefficient in coefficients)
            self.assertEqual(sum((a * b for a, b in zip(coefficients, x)), F(0)), sum(map(abs, coefficients)))

    def test_finite_endpoint_evaluation_independently_matches_mask_proof(self):
        self.assertEqual(max(sum(heads(x), F(0)) for x in vertices(SIZE)), max(all_mask_supports((F(1),) * SIZE)))

    def test_separate_covers_and_joint_cut_have_matching_outer_witnesses(self):
        report = control()['two_cycles']
        self.assertEqual((report['cover_a_floor'], report['cover_b_floor'], report['joint_cut_floor']), ('-1/2', '-1/2', '1/2'))

    def test_unsafe_sibling_has_a_genuine_negative_network_margin(self):
        report = control()['unsafe_sibling']
        x = tuple(map(F, report['actual_input']))
        r = heads(x[:SIZE]) + heads(x[SIZE:])
        self.assertEqual(UNSAFE_BIAS - sum(r), F(-1, 2))
        self.assertEqual(tuple(map(F, report['actual_heads'])), r)

    def test_unsplit_outer_point_is_not_a_sat_network_witness(self):
        self.assertTrue(triangle_valid((F(0),) * SIZE, (F(1),) * SIZE))
        self.assertNotEqual(heads((F(0),) * SIZE), (F(1),) * SIZE)

    def test_partial_duplicate_and_tampered_supports_cannot_authorize_a_cut(self):
        weights = (F(1),) * SIZE
        records = list(enumerate(all_mask_supports(weights)))
        changed = records.copy()
        changed[1] = (1, F(0))
        for bad in (records[:-1], records[:-1] + [records[0]], changed):
            with self.subTest(bad=bad[:2]), self.assertRaises(ValueError):
                complete_cut(weights, bad)

    def test_dyadic_weighted_direction_matches_independent_endpoints(self):
        weights = (F(1, 2), F(1), F(1, 4), F(2), F(1, 2), F(1), F(1, 4))
        endpoint = max(sum((a * b for a, b in zip(weights, heads(x))), F(0)) for x in vertices(SIZE))
        self.assertEqual(endpoint, -complete_cut(weights, list(enumerate(all_mask_supports(weights)))))

    def test_negative_weights_cannot_use_the_positive_relu_mask_identity(self):
        with self.assertRaises(ValueError):
            all_mask_supports((F(-1),) + (F(1),) * (SIZE - 1))

    def test_all_proper_head_hull_mixtures_have_the_claimed_moments(self):
        for mask in range((1 << SIZE) - 1):
            x, opposite = proper_subset_mixture(mask)
            a, b = heads(x), heads(opposite)
            for j in range(SIZE):
                self.assertEqual((x[j] + opposite[j]) / 2, 0)
                if mask & (1 << j):
                    self.assertEqual((a[j] + b[j]) / 2, 1)
        with self.assertRaises(ValueError):
            proper_subset_mixture((1 << SIZE) - 1)

    def test_search_finds_both_components_after_head_and_input_permutations(self):
        rows = two_cycle_rows()
        for shift in (0, 1, 4, 9):
            order = tuple((5 * j + shift) % (2 * SIZE) for j in range(2 * SIZE))
            columns = tuple(reversed(range(2 * SIZE)))
            permuted = [tuple(rows[j][k] for k in columns) for j in order]
            groups = cancellation_components(permuted)
            recovered = {frozenset(order[j] for j in group) for group in groups}
            self.assertEqual(recovered, {frozenset(range(SIZE)), frozenset(range(SIZE, 2 * SIZE))})

    def test_search_must_keep_neutral_prefixes_until_a_complete_support(self):
        # No proper-subset cut separates the mean-x/head1 outer point. Some
        # do improve loose independent head upper bounds; these are different
        # ranking criteria. Requiring positive point violation misses the cycle.
        for mask in range(1, (1 << SIZE) - 1):
            weights = tuple(F(bool(mask & (1 << j))) for j in range(SIZE))
            self.assertGreaterEqual(max(all_mask_supports(weights)), sum(weights))
        self.assertLess(max(all_mask_supports((F(1),) * SIZE)), SIZE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='write deterministic exact report to a NEW file after tests pass')
    parser.add_argument('--verify-report', type=Path, help='read-only replay of a prior deterministic report')
    args = parser.parse_args()
    if args.verify_report:
        verify_report(args.verify_report)
        print('EXACT_REPORT_VERIFIED: no solver, Source or benchmark execution')
        return 0
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExactControl))
    if not result.wasSuccessful():
        return 1
    report = control()
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')
    print(json.dumps({key: report[key] for key in ('schema', 'two_cycles', 'unsafe_sibling', 'benchmark_or_score_claim')}, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
