# -*- coding: utf-8 -*-
"""Primitive invention by anti-unification — the engine invents its own new primitives from recurring
motifs in its library, so even the fixed axiom set expands without a human (owner 2026-07-13)."""
from __future__ import annotations

from atanor_vibecode.abstraction import (
    anti_unify,
    canonical,
    compression_gain,
    holes_in,
    instantiate,
    match,
    mine,
    size,
)
from atanor_vibecode.code_evolver import evaluate, to_source


def _sq_plus_self(v):   # v + v*v
    return ("op", "+", ("var", v), ("op", "*", ("var", v), ("var", v)))


def test_anti_unify_captures_a_repeated_variable_as_one_parameter():
    # a + a*a  and  b + b*b  generalize to  x0 + x0*x0  — ONE hole, not three (the memo table sees the
    # same disagreement (a,b) each time). This is the "square-plus-self" motif, invented, not coded.
    tmpl = canonical(anti_unify(_sq_plus_self("a"), _sq_plus_self("b")))
    assert holes_in(tmpl) == 1
    # instantiate it on a fresh variable and it computes v + v*v
    inst = instantiate(tmpl, [("var", "c")])
    assert evaluate(inst, {"c": 4}) == 4 + 4 * 4 == 20
    assert "hole" not in to_source(inst)                    # instantiation yields an ordinary tree


def test_differing_operators_generalize_to_a_hole():
    t1 = ("op", "+", ("var", "a"), ("var", "b"))
    t2 = ("op", "+", ("var", "a"), ("op", "*", ("var", "b"), ("var", "b")))
    tmpl = canonical(anti_unify(t1, t2))                    # a + <hole>
    assert holes_in(tmpl) == 1
    b = match(tmpl, t2)
    assert b is not None and b[0] == ("op", "*", ("var", "b"), ("var", "b"))


def test_match_enforces_consistent_hole_binding():
    tmpl = canonical(anti_unify(_sq_plus_self("a"), _sq_plus_self("b")))   # x0 + x0*x0
    assert match(tmpl, _sq_plus_self("c")) is not None       # c + c*c  matches
    # c + d*d  must NOT match — hole 0 would have to bind both c and d
    bad = ("op", "+", ("var", "c"), ("op", "*", ("var", "d"), ("var", "d")))
    assert match(tmpl, bad) is None


def test_mine_invents_a_compressing_primitive():
    # a library where the square-plus-self motif recurs across several programs → it gets named.
    lib = [
        _sq_plus_self("a"),
        ("op", "+", _sq_plus_self("b"), ("const", 1)),      # (b + b*b) + 1
        ("op", "*", _sq_plus_self("a"), ("var", "b")),      # (a + a*a) * b
        ("op", "-", ("var", "a"), ("var", "b")),            # noise — shares no motif
    ]
    found = mine(lib, top_k=4, min_gain=2)
    assert found, "expected at least one invented primitive"
    top = found[0]
    assert top["arity"] >= 1 and top["gain"] > 0
    # the top invented primitive is the square-plus-self motif (its body appears 3x)
    assert "x0" in top["source"]
    inst = instantiate(top["template"], [("var", "z")] * top["arity"])
    assert isinstance(evaluate(inst, {"z": 3, "b": 2, "a": 1}), int)   # instantiates to a runnable tree


def test_compression_gain_zero_when_no_recurrence():
    # a motif that appears only once compresses nothing — it must not be named.
    lib = [_sq_plus_self("a"), ("op", "-", ("var", "a"), ("var", "b"))]
    once = canonical(anti_unify(_sq_plus_self("a"), _sq_plus_self("a")))   # identical → no holes
    assert compression_gain(lib, once) == 0                 # holeless / non-recurring → 0


def test_size_and_holes_helpers():
    tmpl = canonical(anti_unify(_sq_plus_self("a"), _sq_plus_self("b")))
    assert size(tmpl) >= 4 and holes_in(tmpl) == 1
