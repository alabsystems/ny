#!/usr/bin/env python3
"""First end-to-end VNN-COMP 2026 scorecard for ny.

`ny_retroactive_scorecard.py` cannot consume the 2026 corpus: that tree uses
VERSIONED subdirectories (`<cat>/1.0/instances.csv`, `<cat>/2.0/...`) and the
scorer asserts exactly one `instances.csv` per category. So until today the
2026 board — half the campaign's stated goal — had never been scored as a whole;
`cifar100_2024`'s 68.8 was computed by hand from the method in its NOTES file.

METHOD (identical to that NOTES file, so the numbers are comparable):
    ny_raw     = 10 * (sat + unsat)                     from reports/measured-2026/<cat>.csv
    winner_raw = max Points column in that category's table in scored.tex
    normalized = 100 * ny_raw / winner_raw, capped at 100

CAVEAT, deliberately not papered over: ny_raw credits every banked sat/unsat as
CORRECT. It is an upper bound. A wrong verdict is -150, and this campaign has
already retracted 21 vit_2023 unsats that an N-ary Concat defect minted, so the
gap between this number and a defended one is exactly the moat.
"""
import argparse, collections, csv, glob, os, re, sys

def winners(tex_path):
    tex = open(tex_path).read()
    out = {}
    for block in re.split(r'\\begin\{table\}', tex):
        m = re.search(r'\\texttt\{([0-9a-z\-]+)', block)
        if not m:
            continue
        pts = [int(x) for x in re.findall(r'&\s*(-?\d+)\s*&\s*[\d.]+\s*&\s*[\d.]+\\%', block)]
        if pts:
            out[m.group(1)] = max(pts)
    return out

def field_totals(tex_path):
    """Each official tool's regular-track total: the sum of its per-category Score column."""
    tot = collections.Counter()
    for block in re.split(r'% Category ', open(tex_path).read())[1:]:
        for tool, score in re.findall(r'\n\d+ & (.+?) & \d+ & \d+ & \d+ & \d+ & -?\d+ & ([\d.]+) &', block):
            tot[tool.replace('$\\alpha$-$\\beta$-CROWN', 'alpha-beta-CROWN')] += float(score)
    return tot

def ny_raw(measured_dir):
    out = {}
    for f in glob.glob(os.path.join(measured_dir, '*.csv')):
        cat = os.path.basename(f)[:-4]
        c = collections.Counter()
        with open(f, newline='') as fh:
            for r in csv.reader(fh):
                if len(r) >= 6:   c[r[4]] += 1
                elif len(r) == 5: c[r[3]] += 1   # off-spec bank: verdict at index 3
        out[cat] = (10 * (c['sat'] + c['unsat']), c)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scored-tex', default=os.path.expanduser('~/vnncomp2026_results/SCORING/latex/scored.tex'))
    ap.add_argument('--measured', default='reports/measured-2026')
    ap.add_argument('--self-check', action='store_true',
                    help='fail loudly unless cifar100_2024 reproduces its published 68.8')
    a = ap.parse_args()
    for p in (a.scored_tex, a.measured):
        if not os.path.exists(p):
            sys.exit(f'path does not resolve: {p}   (an empty result must never look like a zero score)')
    best, ny = winners(a.scored_tex), ny_raw(a.measured)
    if not best: sys.exit('parsed ZERO categories from scored.tex — refusing to report a score')
    if not ny:   sys.exit(f'parsed ZERO banks from {a.measured} — refusing to report a score')

    rows, total, n, unbanked = [], 0.0, 0, []
    for k, w in sorted(best.items()):
        cat = k[5:].replace('-', '_')
        hit = [c for c in ny if '2026-' + c.replace('_', '-') == k]
        if not hit:
            unbanked.append(cat); continue
        raw = ny[hit[0]][0]
        norm = min(100.0, 100.0 * raw / w) if w > 0 else 0.0
        total += norm; n += 1
        rows.append((norm, cat, raw, w))

    print(f'{"category":<44}{"ny_raw":>9}{"winner":>9}{"norm":>8}')
    for norm, cat, raw, w in sorted(rows, key=lambda r: r[0]):
        print(f'{cat[:44]:<44}{raw:>9}{w:>9}{norm:>8.1f}')
    # The denominator is EVERY scored category, banked or not: an unbanked category scores 0,
    # it does not leave the board. (Summing only banked categories reported "of 2200" on a
    # 2400-point board and hid 200 points in vit_2023 + yolo_2023.)
    print(f'\nny 2026 = {total:.1f} of {len(best)*100} ({n} of {len(best)} scored categories banked); mean {total/len(best):.1f}')
    if unbanked:
        print('NO BANK (scores 0): ' + ', '.join(unbanked))
    field = field_totals(a.scored_tex)
    if field:
        print('\nregular-track field (official Score columns; ny inserted by its own total):')
        board = sorted(list(field.items()) + [('ny', total)], key=lambda t: -t[1])
        for i, (tool, sc) in enumerate(board, 1):
            print(f'  {i:>2} {tool:<24}{sc:>8.1f}' + ('   <==' if tool == 'ny' else ''))
        lead = board[0] if board[0][0] != 'ny' else board[1]
        print(f'  gap to first ({lead[0]}): {lead[1] - total:+.1f}   (ny above a winner would lower that'
              ' tool\'s own normalisation; not modelled)')
    print('\nrecoverable, largest first:')
    gaps = [(100 - norm, cat) for norm, cat, raw, w in rows if norm < 100] + [(100.0, c) for c in unbanked]
    for g, cat in sorted(gaps, reverse=True):
        print(f'   {cat[:40]:<40} +{g:5.1f}' + ('   (no bank)' if cat in unbanked else ''))

    if a.self_check:
        c = [r for r in rows if r[1] == 'cifar100_2024']
        if not c: sys.exit('SELF-CHECK FAILED: cifar100_2024 absent')
        if abs(c[0][0] - 68.8) > 0.1:
            sys.exit(f'SELF-CHECK FAILED: cifar100_2024 = {c[0][0]:.1f}, published 68.8')
        print('\nself-check OK: cifar100_2024 reproduces its published 68.8')

if __name__ == '__main__':
    main()
