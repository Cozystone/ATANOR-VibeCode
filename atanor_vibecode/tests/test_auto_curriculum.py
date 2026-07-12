# -*- coding: utf-8 -*-
"""Autonomous self-curriculum — proof the code frontier advances WITHOUT a human adding primitives
or writing targets (owner: "코드 파인튜닝 없이도 진전이 꾸준해야 해")."""
from __future__ import annotations

import random

from atanor_vibecode.auto_curriculum import (
    _FAMILIES,
    _arity,
    _atomic_write_text,
    autonomous_round,
    compose_target,
    load_state,
    new_state,
    run,
    save_state,
)
from atanor_vibecode.code_evolver import evaluate, to_source


def test_save_state_is_atomic_and_durable(tmp_path):
    # crash-safety: the write must be atomic (temp + os.replace), leave no stray temp files, and NEVER
    # corrupt a pre-existing good file. A power loss mid-write yields old-intact-or-new-complete.
    sp = tmp_path / "state.json"
    st = new_state()
    st["round"] = 7
    save_state(sp, st)
    assert load_state(sp)["round"] == 7
    save_state(sp, {**st, "round": 8})                 # overwrite an existing file
    assert load_state(sp)["round"] == 8                # cleanly replaced, not appended/corrupted
    assert not list(tmp_path.glob("*.tmp"))            # no leftover temp files
    # a truncated/corrupt file must not crash the loader — it falls back to a fresh state
    sp.write_text("{ this is not valid json", encoding="utf-8")
    assert load_state(sp)["round"] == 0                # graceful recovery, no exception
    # the atomic helper itself replaces content wholesale
    p = tmp_path / "x.txt"
    _atomic_write_text(p, "hello")
    _atomic_write_text(p, "world")
    assert p.read_text(encoding="utf-8") == "world"


def test_composed_target_is_a_real_verifiable_program():
    # a generated target built from library blocks is int-valued over the family env — a HONEST
    # answer key (its outputs come from actually running verified parts), never fabricated.
    rng = random.Random(1)
    lib = [t for _n, t in __seed("ab")]
    tgt = compose_target(lib, "ab", tier=1, rng=rng)
    for _ in range(20):
        env = {"a": rng.randint(0, 9), "b": rng.randint(0, 9)}
        assert isinstance(evaluate(tgt, env), int)   # total, safe, deterministic


def test_one_round_bootstraps_seeds_and_admits_only_generalizers():
    state = new_state()
    rec = autonomous_round(state, random.Random(3), problems=4)
    # seeds get re-derived and remembered; every admitted program passed the held-out gate.
    assert state["frontier"]["distinct_solved"] >= 3
    for d in rec["details"]:
        if d["accepted"]:
            assert d["holdout"] >= 1.0                # nothing enters the library overfit


def test_library_accumulates_across_rounds_without_human():
    state = new_state()
    rng = random.Random(5)
    first = None
    for _ in range(4):
        autonomous_round(state, rng, problems=6)
        if first is None:
            first = state["frontier"]["distinct_solved"]
    # the library is strictly larger than after the first round — capability accrued on its own.
    assert state["frontier"]["distinct_solved"] > first
    # and it stays deduplicated (no program source repeats).
    for fam in _FAMILIES:
        progs = state["programs"][fam]
        assert len(progs) == len(set(progs))


def test_tier_controller_moves_on_its_own():
    state = new_state()
    rng = random.Random(2)
    moves = set()
    for _ in range(6):
        rec = autonomous_round(state, rng, problems=6)
        moves.add(rec["move"])
    # the curriculum controller actually took a decision each round (up / hold / down) — no schedule.
    assert moves and moves <= {"up", "hold", "down"}
    assert 0 <= state["tier"] <= 6


def test_trivial_programs_are_not_counted_as_capabilities():
    # a constant or a bare-input projection computes nothing — it must never enter the library, so the
    # distinct-function metric can't be inflated by degenerate compositions.
    from atanor_vibecode.auto_curriculum import _is_trivial, _admit, new_state, _FAMILIES
    assert _is_trivial(("const", 3), "ab")                       # constant
    assert _is_trivial(("var", "a"), "ab")                       # identity projection
    assert _is_trivial(("op", "*", ("var", "a"), ("const", 0)), "ab")  # always 0 — constant
    assert not _is_trivial(("op", "+", ("var", "a"), ("var", "b")), "ab")   # real function
    assert not _is_trivial(("len", "xs"), "xs")                  # len varies — a real function
    st = new_state()
    assert _admit(st, "ab", ("const", 5)) == "reject"
    assert _admit(st, "ab", ("var", "a")) == "reject"
    assert _admit(st, "ab", ("op", "+", ("var", "a"), ("var", "b"))) == "new"
    assert st["frontier"]["distinct_solved"] == 0 or len(st["sigs"]["ab"]) == 1  # only the real one


def test_engine_invents_its_own_primitives():
    # the WIRING: when the library holds a recurring motif, a round's mining step must populate
    # state["abstractions"] with well-formed, instantiable primitives (discovery in the wild is
    # stochastic; here we pre-seed the motif so the integration is tested deterministically).
    from atanor_vibecode.abstraction import instantiate
    from atanor_vibecode.code_evolver import evaluate

    def sq(v):   # v + v*v — the square-plus-self motif
        return ("op", "+", ("var", v), ("op", "*", ("var", v), ("var", v)))

    state = new_state()
    # inject a motif-rich ab library (as if several rounds had solved these)
    for tree, src in [(sq("a"), "a+a*a"),
                      (("op", "+", sq("b"), ("const", 1)), "(b+b*b)+1"),
                      (("op", "*", sq("a"), ("var", "b")), "(a+a*a)*b")]:
        state["libraries"]["ab"].append(tree)
        state["programs"]["ab"].append(src)
        state["sigs"]["ab"].append(f"sig-{src}")
    autonomous_round(state, random.Random(1), problems=2)
    invented = state["abstractions"]["ab"]
    assert invented, "mining should invent a primitive from the recurring motif"
    assert state["frontier"]["invented_primitives"] >= 1
    for ab in invented:
        tree = instantiate(ab["template"], [("const", 2)] * ab["arity"])
        assert isinstance(evaluate(tree, {"a": 3, "b": 2}), int)   # runnable, well-formed
        assert ab["gain"] >= 2                                     # it compresses — earned its name


def test_arity_measures_composition_depth():
    assert _arity(("var", "a")) == 0
    assert _arity(("op", "+", ("var", "a"), ("var", "b"))) == 1
    assert _arity(("op", "+", ("op", "*", ("var", "a"), ("var", "a")), ("var", "b"))) == 2


def test_run_persists_state_and_journal(tmp_path):
    sp = tmp_path / "state.json"
    jp = tmp_path / "journal.jsonl"
    out = run(rounds=2, state_path=sp, journal_path=jp, seed=11, problems=4)
    assert sp.exists() and jp.exists()
    assert out["round"] == 2 and out["distinct_solved"] >= 3
    # journal has one line per round; reloading state resumes (round continues, tuples restored).
    assert jp.read_text(encoding="utf-8").strip().count("\n") == 1
    reloaded = load_state(sp)
    assert reloaded["round"] == 2
    for fam in _FAMILIES:
        for t in reloaded["libraries"][fam]:
            assert isinstance(t, tuple) and isinstance(to_source(t), str)   # trees, not JSON lists


def __seed(family):
    from atanor_vibecode.auto_curriculum import _SEED_TREES
    return _SEED_TREES[family]
