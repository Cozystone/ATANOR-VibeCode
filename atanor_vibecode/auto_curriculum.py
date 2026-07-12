# -*- coding: utf-8 -*-
"""Autonomous self-curriculum for the code evolver (owner 2026-07-13:
"음성명령 기반 의도 코딩이 잘 되려면 이런 코드 파인튜닝 없이도 진전이 꾸준해야 해").

Until now the engine's capability jumped only when a HUMAN added a primitive: + - * // (v1),
conditionals (v3), fold/len (v4), map/filter (RSI-6). That hand-tuning is exactly what should NOT
be needed. This module makes the frontier advance on its own.

THE MECHANISM — self-generated, self-verified problems.
Code has a free exact oracle (the interpreter). So the engine can invent a problem it is able to
CHECK without any human answer key: take programs it has ALREADY solved (its library), compose them
with the safe interpreter into a new target, run that composite over sampled inputs to get the
reference outputs, then hide the composite and make the search RE-DERIVE it (with the library
available as building blocks). Every "answer key" is a real verified program actually run — nothing
is fabricated. A solved problem becomes a new building block, so the next round can compose deeper.
A curriculum controller raises difficulty automatically when the solve-rate stays high, and holds
when it collapses. No human writes targets, sets the schedule, or edits a primitive.

CAPABILITY = THE FUNCTION, NOT THE TREE (anti-bloat, honesty).
A naive compose-and-keep loop fills the library with ever-more-baroque monster expressions that
pass verification but teach nothing — fake progress. So the library is keyed by BEHAVIORAL
SIGNATURE (the tuple of outputs over a fixed probe battery), and it keeps the SMALLEST program per
signature. Two syntactically different trees that compute the same function collapse to one capability;
a shorter re-derivation of a known function counts as compression. "distinct_solved" therefore means
distinct FUNCTIONS the engine can compute — an honest measure — and generation is biased toward small
blocks with bounded arity so trees stay compact. Every acceptance still passes a held-out
generalization gate (the synthesize_verified discipline). Nothing is exec'd; trees are interpreted.

HONEST SCOPE: this accumulates COMPOSITIONAL DEPTH over a fixed axiom set — re-deriving and
remembering deeper, compact combinations of its primitives — which is genuine, measurable capability
growth. It does not conjure new mathematics from nothing.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from atanor_vibecode.abstraction import instantiate as _instantiate
from atanor_vibecode.abstraction import mine as _mine_abstractions
from atanor_vibecode.code_evolver import evaluate, evolve, fitness, to_source

# ---------------------------------------------------------------------------
# Families: a problem "family" fixes the input signature so any two solved programs in it compose
# with a consistent env. Every solved tree is INT-valued over its family's env, so composing two of
# them with an arithmetic op is again a valid int-valued program in the same family.
# ---------------------------------------------------------------------------
_FAMILIES: dict[str, dict[str, Any]] = {
    "ab": {"vars_": ["a", "b"], "list_vars": (), "control_flow": True},
    "xs": {"vars_": [], "list_vars": ("xs",), "control_flow": False},
}

# A FIXED probe battery per family — the behavioral fingerprint of a program is its outputs here.
# Two trees with identical fingerprints compute the same function (up to these probes) and are the
# same capability; we keep the smallest. Deterministic, so signatures compare across rounds/runs.
_PROBES: dict[str, list[dict[str, Any]]] = {
    "ab": [{"a": a, "b": b} for a, b in
           [(0, 0), (1, 0), (0, 1), (2, 3), (3, 2), (5, 1), (1, 5), (4, 4), (7, 2), (2, 7), (6, 3), (9, 0)]],
    "xs": [{"xs": L} for L in
           [[], [0], [1], [2, 2], [1, 2, 3], [3, 1, 2], [4, 0, 4], [5, 5, 5, 5], [1, 2, 3, 4], [6, 1], [0, 0, 0], [2, 4, 6]]],
}


def _sample_env(family: str, rng: random.Random) -> dict[str, Any]:
    if family == "ab":
        return {"a": rng.randint(0, 9), "b": rng.randint(0, 9)}
    n = rng.randint(1, 6)
    return {"xs": [rng.randint(0, 7) for _ in range(n)]}


def signature(tree: Any, family: str) -> str:
    """The behavioral fingerprint: outputs over the fixed probe battery, joined to a stable string.
    This is the capability identity — what the program COMPUTES, independent of how it's written."""
    return ",".join(str(evaluate(tree, env)) for env in _PROBES[family])


def _size(tree: Any) -> int:
    """Node count — the parsimony measure. Smaller programs are preferred (compression = understanding)."""
    if not isinstance(tree, (tuple, list)):
        return 1
    if not tree:
        return 1
    return 1 + sum(_size(t) for t in tree[1:] if isinstance(t, (tuple, list)))


def _is_trivial(tree: Any, family: str) -> bool:
    """A program is a genuine capability only if it actually COMPUTES something. Reject the degenerate
    cases so the metric can't be inflated: a constant (same output for every probe) or a pure input
    projection (behaviorally identical to a bare input variable). len/sum/max etc. all survive."""
    outs = [evaluate(tree, env) for env in _PROBES[family]]
    if len(set(outs)) <= 1:
        return True                                          # constant — computes nothing
    for v in _FAMILIES[family]["vars_"]:
        if outs == [env[v] for env in _PROBES[family]]:
            return True                                      # identity projection — computes nothing
    return False


# Seed axioms — the small hand-given starting set. Everything past these is reached by composition,
# not by a human adding a primitive. Each seed is a compact canonical target the engine re-derives.
_SEED_TREES: dict[str, list[tuple[str, Any]]] = {
    "ab": [
        ("a+b", ("op", "+", ("var", "a"), ("var", "b"))),
        ("a*a+b", ("op", "+", ("op", "*", ("var", "a"), ("var", "a")), ("var", "b"))),
        ("max(a,b)", ("if", ("cmp", ">", ("var", "a"), ("var", "b")), ("var", "a"), ("var", "b"))),
    ],
    "xs": [
        ("sum(xs)", ("fold", "+", ("const", 0), "xs")),
        ("len(xs)", ("len", "xs")),
        ("sum_evens", ("fold", "+", ("const", 0),
                       ("filter", ("cmp", "==", ("op", "%", ("var", "_x"), ("const", 2)),
                                   ("const", 0)), "xs"))),
    ],
}

_LIB_CAP = 40            # bounded distinct functions per family
_MAX_KEEP_SIZE = 34      # reject a solution too bloated to be a clean building block
_UP, _DOWN = 0.7, 0.34   # competence thresholds: mastery raises / failure lowers the tier
_SATURATED = 0.2         # novelty below this (at mastery) means the tier is exhausted → climb
_FAST = 0.5              # novelty above this means we're learning fast → push ahead


def _tests_from_tree(tree: Any, family: str, n: int, rng: random.Random
                     ) -> list[tuple[dict[str, Any], int]]:
    """Turn a (verified) target tree into input→output examples by running it — the honest answer key."""
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 6:
        tries += 1
        env = _sample_env(family, rng)
        key = json.dumps(env, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append((env, evaluate(tree, env)))
    return out


def compose_target(library: list[Any], family: str, tier: int, rng: random.Random,
                   abstractions: tuple = ()) -> Any:
    """Invent a NEW target by combining solved building blocks with the safe interpreter. Operands are
    biased toward SMALL blocks and arity is bounded, so targets stay compact (no monster bloat). Higher
    tiers allow one more operand and (for 'ab') a conditional wrap — difficulty grows structurally.
    When the engine has INVENTED primitives (mined abstractions), it sometimes builds the target from
    one — instantiating the motif on fresh operands — so its own inventions drive the next problems."""
    from atanor_vibecode.code_evolver import _OPS, _CMP  # local: whitelisted primitive names

    fam = _FAMILIES[family]
    blocks = sorted(library, key=_size)                      # prefer the compact, canonical blocks
    small = blocks[: max(3, len(blocks) // 2 + 1)] or blocks
    leaves = [("var", v) for v in fam["vars_"]] + [("const", rng.randint(1, 3))]

    def pick() -> Any:
        if small and rng.random() < 0.7:
            return rng.choice(small)
        return rng.choice(leaves) if leaves else rng.choice(small or [("const", 1)])

    # seed the target from an INVENTED primitive — the engine building on its own abstractions.
    if abstractions and rng.random() < 0.4:
        ab = rng.choice(abstractions)
        node = _instantiate(ab["template"], [pick() for _ in range(ab["arity"])])
        if tier >= 1 and rng.random() < 0.5:
            node = ("op", rng.choice(list(_OPS)), node, pick())
        return node

    arity = 2 if tier < 2 else rng.choice([2, 2, 3])
    node = pick()
    for _ in range(arity - 1):
        node = ("op", rng.choice(list(_OPS)), node, pick())
    if family == "ab" and tier >= 1 and rng.random() < 0.3:
        node = ("if", ("cmp", rng.choice(list(_CMP)), ("var", "a"), ("var", "b")), node, pick())
    return node


# ---------------------------------------------------------------------------
# State: libraries[fam] = trees, aligned with programs[fam] = sources and sigs[fam] = fingerprints.
# Dedup by signature keeps the smallest tree per FUNCTION — the library is a set of capabilities.
# ---------------------------------------------------------------------------
def new_state() -> dict[str, Any]:
    return {"round": 0, "tier": 0,
            "libraries": {f: [] for f in _FAMILIES},
            "programs": {f: [] for f in _FAMILIES},
            "sigs": {f: [] for f in _FAMILIES},
            "abstractions": {f: [] for f in _FAMILIES},   # invented primitives (mined motifs)
            "history": [],
            "frontier": {"distinct_solved": 0, "compressions": 0, "avg_size": 0.0,
                         "invented_primitives": 0}}


def _admit(state: dict[str, Any], family: str, tree: Any) -> str:
    """Add a capability by behavioral signature. Returns 'new' (a function not seen before),
    'compressed' (a shorter program for a known function), or 'dup'/'reject'."""
    if _size(tree) > _MAX_KEEP_SIZE or _is_trivial(tree, family):
        return "reject"
    sig = signature(tree, family)
    sigs = state["sigs"][family]
    src = to_source(tree)
    if sig not in sigs:
        if len(sigs) >= _LIB_CAP:
            return "reject"
        state["libraries"][family].append(tree)
        state["programs"][family].append(src)
        sigs.append(sig)
        return "new"
    i = sigs.index(sig)
    if _size(tree) < _size(state["libraries"][family][i]):
        state["libraries"][family][i] = tree                 # compression: keep the shorter program
        state["programs"][family][i] = src
        return "compressed"
    return "dup"


def _solve_and_gate(target: Any, family: str, tier: int, library: list[Any],
                    rng: random.Random) -> dict[str, Any]:
    """Solve a generated target on TRAIN and require it to pass a fresh HOLDOUT (generalization) before
    it may enter the library — the synthesize_verified discipline, so the library stays clean."""
    fam = _FAMILIES[family]
    train = _tests_from_tree(target, family, 14, rng)
    holdout = _tests_from_tree(target, family, 10, rng)
    budget = 90 + 30 * tier
    res = evolve(train, fam["vars_"], list_vars=fam["list_vars"],
                 control_flow=fam["control_flow"], library=tuple(library),
                 pop=110, generations=min(budget, 240), rng_seed=rng.randint(1, 10_000))
    tree = res.get("tree")
    hold = fitness(tree, holdout) if tree else 0.0
    accepted = bool(res["solved"] and hold >= 1.0)
    return {"accepted": accepted, "solved": res["solved"], "holdout": round(hold, 3),
            "program": res["program"], "tree": tree, "target": to_source(target)}


def autonomous_round(state: dict[str, Any], rng: random.Random, *, problems: int = 6) -> dict[str, Any]:
    """One self-driven round: bootstrap missing seeds, then generate + solve composed problems, admit
    the generalizers by signature, and let the controller move the tier. Mutates and returns state."""
    state["round"] += 1
    tier = state["tier"]
    attempts, solved_ok, admitted, compressed, details = 0, 0, 0, 0, []

    for family in _FAMILIES:
        lib = state["libraries"][family]
        # (1) bootstrap the family's compact seed axioms before composing.
        for name, tree in _SEED_TREES[family]:
            if signature(tree, family) in state["sigs"][family]:
                continue
            attempts += 1
            g = _solve_and_gate(tree, family, tier, lib, rng)
            if g["accepted"]:
                solved_ok += 1
                verdict = _admit(state, family, g["tree"])
                admitted += verdict == "new"
                compressed += verdict == "compressed"
            details.append({"family": family, "kind": "seed", "name": name, **_slim(g)})

        # (2) compose: invent new targets from what's solved, re-derive them, keep the generalizers.
        # The engine's own INVENTED primitives (mined motifs) seed some of these targets.
        abns = tuple(state["abstractions"].get(family, ()))
        per_family = max(1, problems // len(_FAMILIES))
        for _ in range(per_family):
            if not lib:
                break
            target = compose_target(lib, family, tier, rng, abstractions=abns)
            attempts += 1
            g = _solve_and_gate(target, family, tier, lib, rng)
            verdict = "reject"
            if g["accepted"]:
                solved_ok += 1
                verdict = _admit(state, family, g["tree"])
                admitted += verdict == "new"
                compressed += verdict == "compressed"
            details.append({"family": family, "kind": "composed", "verdict": verdict,
                            "size": _size(g["tree"]) if g["tree"] else 0, **_slim(g)})

    # (3) INVENT PRIMITIVES: mine recurring motifs from each family library into parameterized
    # abstractions (anti-unification), so the engine expands its own building-block vocabulary — not
    # just composition depth over a fixed axiom set, but new named primitives it discovered for itself.
    invented = 0
    for family in _FAMILIES:
        found = _mine_abstractions(state["libraries"][family], top_k=6, min_gain=2)
        state["abstractions"][family] = [
            {"template": a["template"], "arity": a["arity"], "source": a["source"], "gain": a["gain"]}
            for a in found]
        invented += len(found)

    # Two DIFFERENT signals drive the controller (the earlier single "solve-rate" was backwards —
    # novelty falls as a tier is exhausted, which must trigger a CLIMB, not a stall):
    #   competence = fraction of self-generated targets the engine actually solves+generalizes
    #   novelty    = fraction that were NEW functions (still learning at this tier)
    competence = (solved_ok / attempts) if attempts else 0.0
    novelty = (admitted / attempts) if attempts else 0.0
    moved = "hold"
    if competence >= _UP and novelty < _SATURATED:
        state["tier"] = min(tier + 1, 6)                     # MASTERED + saturated → harder targets
        moved = "up" if state["tier"] != tier else "hold"
    elif novelty >= _FAST:
        state["tier"] = min(tier + 1, 6)                     # learning fast → push ahead
        moved = "up" if state["tier"] != tier else "hold"
    elif competence < _DOWN and tier > 0:
        state["tier"] = tier - 1                             # can't solve its own targets → ease off
        moved = "down"

    distinct = sum(len(s) for s in state["sigs"].values())
    sizes = [_size(t) for f in _FAMILIES for t in state["libraries"][f]]
    state["frontier"] = {"distinct_solved": distinct,
                         "compressions": state["frontier"].get("compressions", 0) + compressed,
                         "avg_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
                         "invented_primitives": invented}
    rec = {"round": state["round"], "tier_before": tier, "tier_after": state["tier"], "move": moved,
           "attempts": attempts, "admitted": admitted, "compressed": compressed,
           "competence": round(competence, 3), "novelty": round(novelty, 3),
           "frontier": dict(state["frontier"]), "details": details, "ts": time.time()}
    state["history"].append({k: rec[k] for k in ("round", "tier_after", "competence", "novelty", "admitted")})
    state["history"] = state["history"][-200:]
    return rec


def _slim(g: dict[str, Any]) -> dict[str, Any]:
    return {"accepted": g["accepted"], "solved": g["solved"], "holdout": g["holdout"],
            "program": g["program"], "target": g["target"]}


def _arity(tree: Any) -> int:
    """How many primitive combinators the tree chains — a cheap structural difficulty proxy."""
    if not isinstance(tree, tuple):
        return 0
    if tree[0] in ("op", "if"):
        return 1 + sum(_arity(t) for t in tree[1:] if isinstance(t, tuple))
    return 0


# ---------------------------------------------------------------------------
# Persistence + bounded daemon
# ---------------------------------------------------------------------------
def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
            base = new_state()
            for k in ("round", "tier"):
                base[k] = s.get(k, base[k])
            for f in _FAMILIES:
                base["libraries"][f] = [_as_tree(t) for t in s.get("libraries", {}).get(f, [])]
                base["programs"][f] = list(s.get("programs", {}).get(f, []))
                base["sigs"][f] = list(s.get("sigs", {}).get(f, []))
                base["abstractions"][f] = [{**a, "template": _as_tree(a.get("template"))}
                                           for a in s.get("abstractions", {}).get(f, [])]
            base["history"] = s.get("history", [])
            base["frontier"] = s.get("frontier", base["frontier"])
            return base
        except Exception:
            pass
    return new_state()


def _as_tree(t: Any) -> Any:
    """JSON round-trips tuples to lists; the interpreter matches on tuples, so restore them."""
    if isinstance(t, list):
        return tuple(_as_tree(x) for x in t)
    return t


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def run(rounds: int = 8, *, state_path: Path | None = None, journal_path: Path | None = None,
        seed: int | None = None, problems: int = 6, log=None) -> dict[str, Any]:
    """Run N self-curriculum rounds, persisting library + tier + a per-round journal so the owner can
    watch capability accrue WITHOUT touching code. Single-writer (this is the only mutator of state)."""
    state_path = state_path or _default_state_path()
    journal_path = journal_path or state_path.with_name("curriculum_journal.jsonl")
    state = load_state(state_path)
    rng = random.Random(seed if seed is not None else (state["round"] * 1009 + int(time.time()) % 9973))
    for _ in range(rounds):
        rec = autonomous_round(state, rng, problems=problems)
        save_state(state_path, state)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if log:
            log(f"[curriculum r{rec['round']}] tier {rec['tier_before']}→{rec['tier_after']} "
                f"({rec['move']}) new {rec['admitted']} + compressed {rec['compressed']} / {rec['attempts']} "
                f"competence={rec['competence']} novelty={rec['novelty']} frontier={rec['frontier']}")
    return {"round": state["round"], "tier": state["tier"], "frontier": state["frontier"],
            "distinct_solved": state["frontier"]["distinct_solved"],
            "libraries": {f: state["programs"][f] for f in _FAMILIES}}


def _default_state_path() -> Path:
    # runtime/, never the source tree — this is generated state, not code.
    return Path(__file__).resolve().parents[1] / "runtime" / "evolution" / "curriculum_state.json"
