#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""NY-native numerical proposals for certified paired-network LP duals.

This is a custom projected Adam optimizer implemented with NumPy numerical
kernels. It does not import or invoke a foreign LP/MIP solver. Its output is a
proposal containing multipliers only, never a verification result. Rust must
rebuild the authenticated source scaffold and certify the resulting residual.

Input grammar (whitespace separated; # comments allowed):
  PAIRED_LP_GEQ_V1 nv ne ni nout
  nv pairs: lower upper
  ne equality rows: pivot nnz (column coefficient)* rhs
  ni inequality rows: nnz (column coefficient)* rhs
  nout pairs: f_output_column g_output_column
All inequalities mean a*x >= rhs. Each source affine equality must have its
unique output as its largest-index column, with coefficient exactly +1. Input
variables and ReLU outputs have no equality pivot and remain boxed variables.

The optimizer eliminates all affine equalities in its proposal objective and
optimizes only nonnegative ReLU multipliers. It recovers free equality
multipliers by reverse substitution against ORIGINAL rows before emission.
The Rust checker encloses residuals for every original boxed variable, so
rounding in elimination and optimization has no verdict authority.
Duplicate columns are numerically coalesced with math.fsum for the proposal;
the authoritative Rust scaffold retains all original coefficients unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


@dataclass
class Row:
    columns: np.ndarray
    coefficients: np.ndarray
    rhs: float
    pivot: int | None = None


@dataclass
class Model:
    lower: np.ndarray
    upper: np.ndarray
    equalities: list[Row]
    inequalities: list[Row]
    outputs: list[tuple[int, int]]
    terms_coalesced_or_zeroed: int = 0


def read_model(path: Path) -> Model:
    words = iter(" ".join(line.split("#", 1)[0] for line in path.read_text().splitlines()).split())

    def token():
        try:
            return next(words)
        except StopIteration as error:
            raise ValueError("truncated paired LP scaffold") from error

    def integer():
        value = int(token())
        if value < 0:
            raise ValueError("negative count or column")
        return value

    def number():
        value = float(token())
        if not math.isfinite(value):
            raise ValueError("nonfinite scaffold number")
        return value

    if token() != "PAIRED_LP_GEQ_V1":
        raise ValueError("expected explicit PAIRED_LP_GEQ_V1 inequality convention")
    nv, ne, ni, nout = [integer() for _ in range(4)]
    if not 0 < nv <= 100_000 or ne > 100_000 or ni > 500_000 or not 0 < nout <= 64:
        raise ValueError("scaffold exceeds proposal resource caps")
    boxes = np.asarray([[number(), number()] for _ in range(nv)], dtype=np.float64)
    if np.any(boxes[:, 0] > boxes[:, 1]):
        raise ValueError("unordered scaffold box")

    terms_coalesced_or_zeroed = 0

    def row(equality=False):
        nonlocal terms_coalesced_or_zeroed
        pivot = integer() if equality else None
        count = integer()
        if count > min(1_000_000, 4 * nv + 16):
            raise ValueError("raw sparse row exceeds proposal resource cap")
        terms = [(integer(), number()) for _ in range(count)]
        rhs = number()
        if any(column >= nv for column, _ in terms):
            raise ValueError("row column out of range")
        grouped = {}
        for column, coefficient in terms:
            grouped.setdefault(column, []).append(coefficient)
        # Only the numerical proposal uses these sums. Its candidate is
        # checked against the unmodified authored row in Rust, where every
        # original coefficient participates in directed residual arithmetic.
        terms = []
        for column in sorted(grouped):
            coefficient = math.fsum(grouped[column])
            if not math.isfinite(coefficient):
                raise ValueError("numerical sparse-column coalescing overflow")
            if coefficient != 0.0:
                terms.append((column, coefficient))
        terms_coalesced_or_zeroed += count - len(terms)
        if equality and (not terms or terms[-1] != (pivot, 1.0)):
            raise ValueError("expected authored +1 output pivot and strictly earlier sources")
        return Row(np.asarray([c for c, _ in terms], dtype=np.int64),
                   np.asarray([v for _, v in terms], dtype=np.float64), rhs, pivot)

    equalities = [row(True) for _ in range(ne)]
    inequalities = [row() for _ in range(ni)]
    outputs = [(integer(), integer()) for _ in range(nout)]
    if any(f >= nv or g >= nv for f, g in outputs):
        raise ValueError("output column out of range")
    if next(words, None) is not None:
        raise ValueError("trailing scaffold tokens")
    return Model(boxes[:, 0], boxes[:, 1], equalities, inequalities, outputs,
                 terms_coalesced_or_zeroed)


class SparseMatrix:
    """Small native sparse matrix using NumPy bincount numerical kernels."""

    def __init__(self, dense):
        self.nrows, self.ncols = dense.shape
        self.rows, self.columns = np.nonzero(dense)
        self.values = dense[self.rows, self.columns]
        self.row_positions = np.r_[0, np.cumsum(np.bincount(self.rows, minlength=self.nrows))]

    def transpose(self, weights):
        return np.asarray([np.bincount(self.columns,
            weights=self.values * row[self.rows], minlength=self.ncols) for row in weights])

    def multiply(self, values):
        return np.asarray([np.bincount(self.rows,
            weights=self.values * row[self.columns], minlength=self.nrows) for row in values])

    def row(self, index):
        start, end = self.row_positions[index:index + 2]
        return self.columns[start:end], self.values[start:end]


def eliminate_affines(model: Model, deadline: float):
    """Proposal-only symbolic affine expressions over input/ReLU variables."""
    nv = len(model.lower)
    pivots = {}
    for index, row in enumerate(model.equalities):
        if not len(row.columns):
            raise ValueError("affine equality has no output pivot")
        target = row.pivot
        if target is None or target != int(row.columns[-1]) or row.coefficients[-1] != 1.0 or target in pivots:
            raise ValueError("expected unique authored +1 affine output pivot")
        pivots[target] = index
    free = np.asarray([j for j in range(nv) if j not in pivots], dtype=np.int64)
    if nv * len(free) > 40_000_000:
        raise ValueError("affine elimination exceeds 320 MB resource cap")
    transform = np.zeros((nv, len(free)), dtype=np.float64)
    transform[free, np.arange(len(free))] = 1.0
    offset = np.zeros(nv, dtype=np.float64)
    ordered = sorted(pivots.items())
    for target, index in ordered:
        if time.monotonic() >= deadline:
            raise TimeoutError("proposal deadline during affine elimination")
        row = model.equalities[index]
        offset[target] = row.rhs
        for source, coefficient in zip(row.columns[:-1], row.coefficients[:-1]):
            transform[target] -= coefficient * transform[source]
            offset[target] -= coefficient * offset[source]
    if not np.all(np.isfinite(transform)) or not np.all(np.isfinite(offset)):
        raise ValueError("numerical affine elimination overflow")
    if len(model.inequalities) * len(free) > 40_000_000:
        raise ValueError("reduced rows exceed 320 MB resource cap")
    reduced = np.zeros((len(model.inequalities), len(free)), dtype=np.float64)
    rhs = np.zeros(len(model.inequalities), dtype=np.float64)
    for j, row in enumerate(model.inequalities):
        if j % 128 == 0 and time.monotonic() >= deadline:
            raise TimeoutError("proposal deadline during row elimination")
        rhs[j] = row.rhs
        for column, coefficient in zip(row.columns, row.coefficients):
            reduced[j] += coefficient * transform[column]
            rhs[j] -= coefficient * offset[column]
    objective = []
    constant = []
    labels = []
    for output, (f, g) in enumerate(model.outputs):
        for sign in [1, -1]:
            objective.append(sign * (transform[f] - transform[g]))
            constant.append(sign * (offset[f] - offset[g]))
            labels.append((output, sign))
    return free, reduced, rhs, np.asarray(objective), np.asarray(constant), labels, ordered


def dual_values(matrix, rhs, objective, constant, weights):
    residual = objective - matrix.transpose(weights)
    values = constant + weights @ rhs - np.sum(np.abs(residual), axis=1)
    return values, residual


def backward_seed(matrix, rhs, objective, constant, deadline):
    """Native greedy activation supports after exact-source affine elimination.

    Only inequalities whose largest free column is the current activation are
    used. Thus choosing a plane never reintroduces a previously eliminated
    later activation. Every proposed weight remains nonnegative.
    """
    weights = np.zeros((len(objective), matrix.nrows), dtype=np.float64)
    residual = objective.copy()
    linear = constant.copy()
    by_pivot = [[] for _ in range(matrix.ncols)]
    for row in range(matrix.nrows):
        columns, coefficients = matrix.row(row)
        if len(columns):
            by_pivot[int(columns[-1])].append((row, columns, coefficients))
    for target in range(matrix.ncols - 1, -1, -1):
        if time.monotonic() >= deadline:
            break
        for output in range(len(objective)):
            current = linear[output] - np.sum(np.abs(residual[output]))
            best = None
            for row, columns, coefficients in by_pivot[target]:
                multiplier = residual[output, target] / coefficients[-1]
                if not math.isfinite(multiplier) or multiplier <= 0.0:
                    continue
                proposal = residual[output, columns] - multiplier * coefficients
                value = current + multiplier * rhs[row] + np.sum(np.abs(residual[output, columns])) - np.sum(np.abs(proposal))
                if value >= current and (best is None or value > best[0]):
                    best = (value, row, multiplier, columns, coefficients)
            if best is not None:
                _, row, multiplier, columns, coefficients = best
                weights[output, row] += multiplier
                residual[output, columns] -= multiplier * coefficients
                linear[output] += multiplier * rhs[row]
    return weights


def recover_equalities(model, ordered, objective, inequalities):
    residual = objective.copy()
    for multiplier, row in zip(inequalities, model.inequalities):
        if multiplier != 0.0:
            residual[row.columns] -= multiplier * row.coefficients
    equalities = np.zeros(len(model.equalities), dtype=np.float64)
    for target, index in reversed(ordered):
        row = model.equalities[index]
        multiplier = residual[target]  # authored target coefficient is +1
        equalities[index] = multiplier
        residual[row.columns] -= multiplier * row.coefficients
    return equalities


def optimize(model: Model, args, deadline):
    free, rows, rhs, c, constant, labels, ordered = eliminate_affines(model, deadline)
    center = model.lower[free] * 0.5 + model.upper[free] * 0.5
    radius = model.upper[free] * 0.5 - model.lower[free] * 0.5
    constant = constant + c @ center
    rhs = rhs - rows @ center
    rows *= radius[None, :]
    c *= radius[None, :]
    scales = np.maximum(np.maximum(np.sum(np.abs(rows), axis=1), np.abs(rhs)), 1e-12)
    rows /= scales[:, None]
    rhs /= scales
    if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(c)):
        raise ValueError("reduced proposal matrix overflow")
    matrix = SparseMatrix(rows)
    del rows
    weights = backward_seed(matrix, rhs, c, constant, deadline)
    best_weights = weights.copy()
    best_values, _ = dual_values(matrix, rhs, c, constant, weights)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    learning = args.learning_rate * np.maximum(1.0, np.max(np.abs(c), axis=1))
    step = 0
    history = []
    while matrix.nrows > 0 and step < args.steps and time.monotonic() < deadline:
        values, residual = dual_values(matrix, rhs, c, constant, weights)
        improved = values > best_values
        best_values[improved] = values[improved]
        best_weights[improved] = weights[improved]
        if step % args.record_every == 0:
            history.append({"step": step, "proposal_lower_bounds": best_values.tolist()})
        # Smooth only the proposal gradient; retain the actual nonsmooth dual
        # value when choosing candidates. No smooth score is a certificate.
        smoothing = np.maximum(1e-10, np.max(np.abs(c), axis=1) * 1e-3 * args.anneal ** step)
        derivative = residual / np.hypot(residual, smoothing[:, None])
        gradient = rhs[None, :] + matrix.multiply(derivative)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        step += 1
        direction = (first_moment / (1.0 - 0.9 ** step)) / (np.sqrt(second_moment / (1.0 - 0.999 ** step)) + 1e-12)
        weights = np.maximum(0.0, weights + (learning * args.anneal ** step)[:, None] * direction)
    values, _ = dual_values(matrix, rhs, c, constant, weights)
    improved = values > best_values
    best_values[improved] = values[improved]
    best_weights[improved] = weights[improved]
    candidates = []
    for index, (output, sign) in enumerate(labels):
        inequality = best_weights[index] / scales
        objective = np.zeros(len(model.lower), dtype=np.float64)
        f, g = model.outputs[output]
        objective[f] += sign
        objective[g] -= sign
        equality = recover_equalities(model, ordered, objective, inequality)
        if not np.all(np.isfinite(equality)) or not np.all(np.isfinite(inequality)):
            raise ValueError("nonfinite proposed multiplier")
        candidates.append({"output": output, "sign": sign,
            "inequality_multipliers": inequality.tolist(),
            "equality_multipliers": equality.tolist(),
            "proposal_lower_bound": float(best_values[index])})
    return {"schema": "paired_lp_dual_candidate_v1", "constraint_sense": "geq_rhs",
        "authority": "numerical_proposal_only", "iterations": step,
        "proposal_terms_coalesced_or_zeroed": model.terms_coalesced_or_zeroed,
        "free_variables": len(free), "reduced_nonzeros": len(matrix.values),
        "candidates": candidates, "history": history}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--certificate", type=Path,
        help="multiplier-only Rust input; defaults to --out with .dual suffix")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--anneal", type=float, default=0.995)
    parser.add_argument("--record-every", type=int, default=100)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.seconds) or args.seconds <= 0 or args.steps < 0
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
        or not 0 < args.anneal <= 1 or args.record_every <= 0):
        parser.error("invalid native proposal budget/optimizer parameter")
    started = time.monotonic()
    model = read_model(args.model)
    result = optimize(model, args, started + args.seconds)
    result["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    result["optimizer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["elapsed_seconds"] = time.monotonic() - started
    certificate = args.certificate or args.out.with_suffix(".dual")
    values = [v for candidate in result["candidates"]
        for key in ["equality_multipliers", "inequality_multipliers"] for v in candidate[key]]
    if any(not math.isfinite(v) or abs(v) > 1e100 for v in values):
        raise ValueError("candidate exceeds native checker multiplier cap")
    lines = [f"PAIRED_LP_DUAL_GEQ_V1 {len(model.equalities)} {len(model.inequalities)} {len(model.outputs)}"]
    for candidate in result["candidates"]:
        for key in ["equality_multipliers", "inequality_multipliers"]:
            lines.append(" ".join(repr(v) for v in candidate[key]))
    certificate.parent.mkdir(parents=True, exist_ok=True)
    certificate.write_text("\n".join(lines) + "\n")
    result["certificate_sha256"] = hashlib.sha256(certificate.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"authority": result["authority"], "iterations": result["iterations"],
        "elapsed_seconds": result["elapsed_seconds"],
        "proposal_lower_bounds": [c["proposal_lower_bound"] for c in result["candidates"]]}, allow_nan=False))


if __name__ == "__main__":
    main()
