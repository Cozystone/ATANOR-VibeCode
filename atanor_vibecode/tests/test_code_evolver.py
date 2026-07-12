# -*- coding: utf-8 -*-
"""Code evolver — proof that No-LLM, gradient-free program search discovers functions from a
verifier alone (refutes 'you must borrow a neural core to learn code')."""
from __future__ import annotations

import itertools

from atanor_vibecode.code_evolver import evaluate, evolve, fitness, mutate, random_tree
import random


def _tests(fn):
    return [({"a": a, "b": b}, fn(a, b)) for a, b in itertools.product(range(6), range(6))]


def test_discovers_addition_from_examples_only():
    out = evolve(_tests(lambda a, b: a + b), ["a", "b"], pop=80, generations=120)
    assert out["solved"] and out["fitness"] == 1.0 and out["program"]


def test_discovers_a_nonlinear_target():
    out = evolve(_tests(lambda a, b: a * a + b), ["a", "b"], pop=80, generations=150)
    assert out["solved"], out["program"]


def test_fitness_is_the_verifier_fraction():
    tests = _tests(lambda a, b: a + b)
    assert fitness(("op", "+", ("var", "a"), ("var", "b")), tests) == 1.0
    assert fitness(("op", "-", ("var", "a"), ("var", "b")), tests) < 1.0


def test_evaluation_is_pure_arithmetic_and_safe():
    # division/modulo by zero degrade to 0, never raise — the interpreter is total
    assert evaluate(("op", "//", ("var", "a"), ("const", 0)), {"a": 5}) == 0
    assert evaluate(("op", "%", ("var", "a"), ("const", 0)), {"a": 5}) == 0


def test_mutation_keeps_a_valid_tree():
    rng = random.Random(3)
    t = random_tree(["a", "b"], rng, depth=3)
    for _ in range(50):
        t = mutate(t, ["a", "b"], rng)
        assert isinstance(evaluate(t, {"a": 2, "b": 3}), int)  # always evaluable


def test_control_flow_discovers_a_conditional_function():
    # max(a,b) needs a branch — a conditional IS the discontinuity, discovered gradient-free
    out = evolve(_tests(lambda a, b: max(a, b)), ["a", "b"], pop=140, generations=220,
                 control_flow=True)
    assert out["solved"], out["program"]


def test_conditional_evaluation_and_source():
    from atanor_vibecode.code_evolver import evaluate as ev, to_source
    tree = ("if", ("cmp", ">", ("var", "a"), ("var", "b")), ("var", "a"), ("var", "b"))
    assert ev(tree, {"a": 5, "b": 2}) == 5 and ev(tree, {"a": 1, "b": 9}) == 9
    assert "if" in to_source(tree)


def test_isolation_gate_rejects_overfit():
    from atanor_vibecode.code_evolver import synthesize_verified
    # train on a single point (a=2 -> 4): a constant/degenerate program fits train but NOT holdout
    train = [({"a": 2, "b": 0}, 4)]
    holdout = [({"a": 3, "b": 0}, 9), ({"a": 5, "b": 0}, 25)]  # target a*a — 1 point can't pin it
    v = synthesize_verified(train, holdout, ["a", "b"], pop=40, generations=40)
    assert v["accepted"] is False and v["holdout_fitness"] < 1.0


def test_isolation_gate_accepts_a_generalizer():
    from atanor_vibecode.code_evolver import synthesize_verified
    import itertools
    grid = list(itertools.product(range(6), range(6)))
    train = [({"a": a, "b": b}, a + b) for a, b in grid if (a + b) % 2 == 0]
    holdout = [({"a": a, "b": b}, a + b) for a, b in grid if (a + b) % 2 == 1]
    v = synthesize_verified(train, holdout, ["a", "b"], pop=80, generations=120)
    assert v["accepted"] is True and v["holdout_fitness"] == 1.0  # a+b generalizes


def test_discovers_a_program_over_a_data_structure():
    # the leap past scalars: sum(xs) is discovered as fold('+', 0, xs) from list→number examples
    rng = random.Random(1)
    lists = [[rng.randint(0, 6) for _ in range(rng.randint(1, 5))] for _ in range(24)]
    tests = [({"xs": L}, sum(L)) for L in lists]
    out = evolve(tests, [], list_vars=("xs",), pop=120, generations=200)
    assert out["solved"], out["program"]
    assert "fold" in out["program"]


def test_fold_and_len_evaluate_over_lists():
    from atanor_vibecode.code_evolver import evaluate as ev
    assert ev(("fold", "+", ("const", 0), "xs"), {"xs": [1, 2, 3, 4]}) == 10   # sum
    assert ev(("fold", "*", ("const", 1), "xs"), {"xs": [2, 3, 4]}) == 24       # product
    assert ev(("len", "xs"), {"xs": [5, 5, 5]}) == 3
    assert ev(("fold", "+", ("const", 0), "xs"), {"xs": []}) == 0               # empty is total


def test_graded_fitness_rewards_near_misses():
    from atanor_vibecode.code_evolver import graded_fitness, fitness
    tests = [({"a": a, "b": b}, a + b) for a, b in itertools.product(range(4), range(4))]
    exact = ("op", "+", ("var", "a"), ("var", "b"))          # correct
    close = ("op", "+", ("var", "a"), ("const", 1))          # a+1: wrong but often near a+b
    far = ("op", "*", ("var", "a"), ("const", 5))            # a*5: usually far
    # exact fitness can't tell close from far when both miss; graded ranks close > far
    assert fitness(close, tests) < 1.0
    assert graded_fitness(close, tests) > graded_fitness(far, tests)
    assert graded_fitness(exact, tests) == 1.0


def test_library_curriculum_accumulates_and_reuses():
    # the self-improving mechanism: a curriculum solves each problem and remembers it as a reusable
    # building block, so the library grows — later problems can compose earlier solutions.
    from atanor_vibecode.code_evolver import evolve_with_library
    rng = random.Random(1)
    lists = [[rng.randint(0, 6) for _ in range(rng.randint(1, 5))] for _ in range(30)]
    curr = [
        {"name": "sum", "tests": [({"xs": x}, sum(x)) for x in lists], "list_vars": ("xs",)},
        {"name": "len", "tests": [({"xs": x}, len(x)) for x in lists], "list_vars": ("xs",)},
        {"name": "sum+len", "tests": [({"xs": x}, sum(x) + len(x)) for x in lists], "list_vars": ("xs",)},
    ]
    res = evolve_with_library(curr, pop=120, generations=200)
    assert res["solved_count"] == 3 and res["library_size"] == 3  # all solved, all remembered


def test_higher_order_map_filter_over_collections():
    # RSI 6: sum of evens is a filter+fold — a higher-order program discovered gradient-free
    from atanor_vibecode.code_evolver import evaluate as ev, evolve
    # unit: the interpreter runs map/filter correctly
    assert ev(("fold", "+", ("const", 0), ("map", ("op", "*", ("var", "_x"), ("var", "_x")), "xs")),
              {"xs": [1, 2, 3]}) == 14                                   # sum of squares
    assert ev(("fold", "+", ("const", 0),
               ("filter", ("cmp", "==", ("op", "%", ("var", "_x"), ("const", 2)), ("const", 0)), "xs")),
              {"xs": [1, 2, 3, 4]}) == 6                                 # sum of evens
    rng = random.Random(2)
    lists = [[rng.randint(0, 7) for _ in range(rng.randint(2, 6))] for _ in range(30)]
    tests = [({"xs": L}, sum(x for x in L if x % 2 == 0)) for L in lists]
    out = evolve(tests, [], list_vars=("xs",), pop=200, generations=300)
    assert out["solved"], out["program"]
