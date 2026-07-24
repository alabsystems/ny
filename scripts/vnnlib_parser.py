# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""VNN-LIB output property parser and evaluator.

Parses output-only relational constraints from VNN-LIB files with at most
two layers of and/or nesting. Fails closed on unsupported syntax.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


class VnnlibParseError(Exception):
    """Raised when VNN-LIB syntax is unsupported or malformed."""


@dataclass
class RelOp:
    """A relational constraint: lhs op rhs."""
    lhs: str
    op: str   # "<=", ">=", "<", ">"
    rhs: str


@dataclass
class Conjunction:
    """AND of relational ops."""
    clauses: list[RelOp]


@dataclass
class OutputProperty:
    """Parsed VNN-LIB output property: conjunction or disjunction of conjunctions."""
    disjunctive: bool
    branches: list[Conjunction]
    output_vars: dict[str, int]


# -- S-expression tokenizer/parser --

def _tokenize(text: str) -> list[str]:
    text = re.sub(r";[^\n]*", "", text)
    text = text.replace("(", " ( ").replace(")", " ) ")
    return text.split()


def _parse_sexpr(tokens: list[str], pos: int) -> tuple[object, int]:
    if pos >= len(tokens):
        raise VnnlibParseError("unexpected end of tokens")
    tok = tokens[pos]
    if tok == "(":
        pos += 1
        items: list[object] = []
        while pos < len(tokens) and tokens[pos] != ")":
            item, pos = _parse_sexpr(tokens, pos)
            items.append(item)
        if pos >= len(tokens):
            raise VnnlibParseError("unmatched (")
        return items, pos + 1
    elif tok == ")":
        raise VnnlibParseError("unexpected )")
    else:
        return tok, pos + 1


def _parse_all_sexprs(text: str) -> list[object]:
    tokens = _tokenize(text)
    results = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse_sexpr(tokens, pos)
        results.append(expr)
    return results


# -- Predicate helpers --

def _is_output_var(name: str) -> bool:
    return bool(re.match(r"^Y_\d+$", name))


def _is_input_var(name: str) -> bool:
    return bool(re.match(r"^X_\d+$", name))


def _extract_relop(expr: list) -> Optional[RelOp]:
    if not isinstance(expr, list) or len(expr) != 3:
        return None
    if expr[0] not in ("<=", ">=", "<", ">"):
        return None
    return RelOp(lhs=str(expr[1]), op=str(expr[0]), rhs=str(expr[2]))


def _is_output_relop(relop: RelOp) -> bool:
    has_output = _is_output_var(relop.lhs) or _is_output_var(relop.rhs)
    has_input = _is_input_var(relop.lhs) or _is_input_var(relop.rhs)
    return has_output and not has_input


def _extract_conjunction(expr: list) -> Optional[Conjunction]:
    if not isinstance(expr, list) or len(expr) < 2 or expr[0] != "and":
        return None
    clauses = []
    for sub in expr[1:]:
        relop = _extract_relop(sub)
        if relop is None:
            return None
        clauses.append(relop)
    return Conjunction(clauses=clauses)


# -- Assertion collectors --

def _collect_output_vars(sexprs: list[object]) -> dict[str, int]:
    output_vars: dict[str, int] = {}
    for expr in sexprs:
        if isinstance(expr, list) and len(expr) == 3 and expr[0] == "declare-const":
            name = str(expr[1])
            if _is_output_var(name):
                output_vars[name] = int(name.split("_")[1])
    return output_vars


def _collect_disjunction_branches(
    body: list,
) -> list[Conjunction]:
    branches: list[Conjunction] = []
    for branch in body[1:]:
        conj = _extract_conjunction(branch)
        if conj is not None:
            out_clauses = [c for c in conj.clauses if _is_output_relop(c)]
            if out_clauses:
                branches.append(Conjunction(clauses=out_clauses))
        else:
            relop = _extract_relop(branch)
            if relop is not None and _is_output_relop(relop):
                branches.append(Conjunction(clauses=[relop]))
            else:
                raise VnnlibParseError(f"unsupported disjunction branch: {branch}")
    return branches


def _collect_output_assertions(
    sexprs: list[object],
) -> tuple[list[RelOp], list[Conjunction], bool]:
    """Return (flat_relops, conjunctions, is_disjunctive)."""
    flat_relops: list[RelOp] = []
    conjunctions: list[Conjunction] = []
    disjunctive = False

    for expr in sexprs:
        if not isinstance(expr, list) or len(expr) != 2 or expr[0] != "assert":
            continue
        body = expr[1]
        if not isinstance(body, list) or len(body) < 2:
            continue

        relop = _extract_relop(body)
        if relop is not None:
            if _is_output_relop(relop):
                flat_relops.append(relop)
            continue

        head = body[0]
        if head == "and":
            conj = _extract_conjunction(body)
            if conj is not None:
                out_clauses = [c for c in conj.clauses if _is_output_relop(c)]
                if out_clauses:
                    conjunctions.append(Conjunction(clauses=out_clauses))
            continue

        if head == "or":
            disjunctive = True
            conjunctions.extend(_collect_disjunction_branches(body))
            continue

    return flat_relops, conjunctions, disjunctive


# -- Public API --

def parse_vnnlib_output_property(text: str) -> OutputProperty:
    """Parse output constraints from a VNN-LIB file.

    Supports flat conjunctive, (assert (and ...)), and (assert (or (and ...) ...)).
    Raises VnnlibParseError on unsupported syntax.
    """
    sexprs = _parse_all_sexprs(text)
    output_vars = _collect_output_vars(sexprs)
    if not output_vars:
        raise VnnlibParseError("no output variables (Y_i) declared")

    flat_relops, conjunctions, disjunctive = _collect_output_assertions(sexprs)

    if flat_relops and not disjunctive:
        combined = Conjunction(clauses=flat_relops)
        if conjunctions:
            combined.clauses.extend(c for conj in conjunctions for c in conj.clauses)
            conjunctions = [combined]
        else:
            conjunctions = [combined]

    if not conjunctions:
        raise VnnlibParseError("no output constraints found in VNN-LIB file")

    return OutputProperty(
        disjunctive=disjunctive, branches=conjunctions, output_vars=output_vars,
    )


def _resolve_value(
    name: str, output_vars: dict[str, int], outputs: list[float],
) -> float:
    if name in output_vars:
        return outputs[output_vars[name]]
    try:
        return float(name)
    except ValueError:
        raise VnnlibParseError(f"cannot resolve '{name}' to a value")


def _eval_relop(
    relop: RelOp, output_vars: dict[str, int], outputs: list[float],
) -> bool:
    lhs = _resolve_value(relop.lhs, output_vars, outputs)
    rhs = _resolve_value(relop.rhs, output_vars, outputs)
    ops = {"<=": lhs <= rhs, ">=": lhs >= rhs, "<": lhs < rhs, ">": lhs > rhs}
    if relop.op not in ops:
        raise VnnlibParseError(f"unknown operator: {relop.op}")
    return ops[relop.op]


def evaluate_output_property(prop: OutputProperty, outputs: list[float]) -> bool:
    """Evaluate whether outputs satisfy the property (True = counterexample confirmed)."""
    def eval_conj(conj: Conjunction) -> bool:
        return all(_eval_relop(c, prop.output_vars, outputs) for c in conj.clauses)

    if prop.disjunctive:
        return any(eval_conj(b) for b in prop.branches)
    return all(eval_conj(b) for b in prop.branches)
