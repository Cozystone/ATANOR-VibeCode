# -*- coding: utf-8 -*-
"""RSI layer 7 — local variable binding (let). Compute a value ONCE, name it, reuse it; the search can
FACTOR a repeated subexpression into a let, so a motif is discovered once, not rediscovered per use."""
from __future__ import annotations

import itertools
import random

from atanor_vibecode.code_evolver import (
    evaluate,
    evolve,
    factor_let,
    mutate,
    random_tree,
    to_source,
)


def test_let_evaluates_and_reuses_a_bound_value():
    # let t = a+b in t*t  ==  (a+b)^2
    tree = ("let", "_t0", ("op", "+", ("var", "a"), ("var", "b")),
            ("op", "*", ("var", "_t0"), ("var", "_t0")))
    for a, b in itertools.product(range(6), range(6)):
        assert evaluate(tree, {"a": a, "b": b}) == (a + b) ** 2
    assert "let" in to_source(tree) and "_t0" in to_source(tree)


def test_scope_is_isolated_no_leak():
    # a temp referenced OUTSIDE its let has no binding — it must read 0, never an outer/sibling value.
    leaky = ("op", "+", ("var", "_t0"),
             ("let", "_t0", ("const", 9), ("var", "_t0")))     # (_t0)  +  (let _t0=9 in _t0)
    # outer _t0 is unbound -> 0 ; inner let binds _t0=9 -> 9 ; total 9 (no leak from inner to outer)
    assert evaluate(leaky, {}) == 9
    # nested lets shadow correctly
    nested = ("let", "_t0", ("const", 2),
              ("let", "_t0", ("const", 5), ("op", "*", ("var", "_t0"), ("var", "_t0"))))
    assert evaluate(nested, {}) == 25                          # inner 5 shadows outer 2


def test_factor_let_names_a_repeated_subexpression():
    rng = random.Random(0)
    # (a+b) * (a+b) — the sum appears twice
    ab = ("op", "+", ("var", "a"), ("var", "b"))
    tree = ("op", "*", ab, ab)
    factored = factor_let(tree, rng)
    assert factored[0] == "let"                                # it introduced a let
    assert factored[2] == ab                                   # the bound value IS the repeated subexpr
    # and it computes exactly the same function
    for a, b in itertools.product(range(5), range(5)):
        assert evaluate(factored, {"a": a, "b": b}) == evaluate(tree, {"a": a, "b": b})


def test_factor_let_noop_when_nothing_repeats():
    rng = random.Random(0)
    tree = ("op", "+", ("var", "a"), ("var", "b"))             # nothing occurs twice
    assert factor_let(tree, rng) is tree


def test_mutation_with_let_stays_evaluable():
    rng = random.Random(4)
    t = random_tree(["a", "b"], rng, depth=3, let_binding=True)
    for _ in range(60):
        t = mutate(t, ["a", "b"], rng, let_binding=True)
        assert isinstance(evaluate(t, {"a": 2, "b": 3}), int)  # always total, never raises


def test_evolve_with_let_solves_square_of_sum():
    # (a+b)^2 — a target whose natural program reuses (a+b) twice; let makes that reuse first-class.
    tests = [({"a": a, "b": b}, (a + b) ** 2) for a, b in itertools.product(range(6), range(6))]
    out = evolve(tests, ["a", "b"], pop=160, generations=200, let_binding=True)
    assert out["solved"], out["program"]


def test_let_binding_is_backward_compatible_off_by_default():
    # with let_binding off (default), no let nodes appear and plain search still works.
    out = evolve([({"a": a, "b": b}, a + b) for a in range(6) for b in range(6)], ["a", "b"])
    assert out["solved"] and "let" not in (out["program"] or "")
