# -*- coding: utf-8 -*-
"""Code evolver — PROOF that No-LLM trial-and-error code learning does NOT need a gradient
(owner 2026-07-12: "이 한계 극복할 방법 없어?").

Gemini's claim: because code is discontinuous (one typo → Fail), you can't count frequencies like
language; you need a differentiable neural core to compute which direction to fix. That is FALSE —
overstated. The thing language lacks and code HAS is a free, exact VERIFIER: the interpreter/tests
return crisp ground truth (pass/fail) plus a symbolic pointer to the fault (the error). That
replaces the gradient. Verifier-guided search over programs — genetic programming, MCTS, type-
directed enumeration, CEGIS — is a decades-old paradigm that learns programs with NO gradient.

This module is the smallest honest demonstration: a population of expression TREES evolves, judged
ONLY by how many input→output examples it satisfies (the verifier), until it discovers the target
function. No neural net, no gradient, no LLM — the fitness IS the compiler. It's the same arena we
already run (population + elitism + antibodies), with the genome being a program and the fitness
being the oracle the interpreter hands us for free.

SAFETY: the genome is a whitelisted arithmetic tree (vars, small int constants, + - * // %) and is
INTERPRETED directly here — never compiled or exec'd — so an evolved candidate can only ever do
arithmetic on the given numbers. This is the safe kernel; a richer engine (control flow, calls)
belongs in the ISOLATED plugin sandbox Gemini rightly described, still gated by critic_integrity +
frozen_oracle before anything reaches the fortress.
"""
from __future__ import annotations

import operator
import random
from typing import Any, Callable

# The whitelisted primitive set — safe because it is INTERPRETED, not exec'd.
_OPS: dict[str, Callable[[int, int], int]] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "//": lambda a, b: a // b if b != 0 else 0, "%": lambda a, b: a % b if b != 0 else 0,
}
# Comparisons for CONTROL FLOW — this is where real code begins: a conditional IS the discontinuity
# Gemini worried about, and the verifier still crosses it gradient-free (a wrong branch fails an
# example, which selection follows).
_CMP: dict[str, Callable[[int, int], bool]] = {
    "<": operator.lt, "<=": operator.le, "==": operator.eq, ">": operator.gt, ">=": operator.ge,
}


def evaluate(tree: Any, env: dict[str, Any]) -> int:
    """Interpret an expression tree over env. Arithmetic + conditionals + a bounded FOLD over an
    input list (this is the leap past scalars into data-structure programs — sum/product/count are
    folds). No exec, no imports, no calls, total (division/empty/comparison never raise)."""
    kind = tree[0]
    if kind == "var":
        return int(env.get(tree[1], 0))
    if kind == "const":
        return int(tree[1])
    if kind == "len":
        return len(evaluate_list(tree[1], env))
    if kind == "fold":
        _, op, init_t, src = tree
        acc = evaluate(init_t, env)
        for x in evaluate_list(src, env):
            try:
                acc = int(_OPS[op](acc, int(x)))
            except Exception:
                pass
        return int(acc)
    if kind == "if":
        _, cond, then_t, else_t = tree
        return evaluate(then_t, env) if _eval_cond(cond, env) else evaluate(else_t, env)
    if kind == "let":
        # local variable binding (RSI layer 7): compute a value ONCE, name it, reuse it in the body.
        # Scope is lexical + isolated: the bound name lives only inside `body` — a reference to it
        # anywhere else sees no binding (env.get default 0), so temp vars never leak out of their let.
        _, name, val_t, body_t = tree
        return evaluate(body_t, {**env, name: evaluate(val_t, env)})
    _, op, left, right = tree
    try:
        return int(_OPS[op](evaluate(left, env), evaluate(right, env)))
    except Exception:
        return 0


def _eval_cond(cond: Any, env: dict[str, Any]) -> bool:
    try:
        _, op, left, right = cond
        return bool(_CMP[op](evaluate(left, env), evaluate(right, env)))
    except Exception:
        return False


_ELEM = "_x"   # the element variable bound inside map/filter bodies


def evaluate_list(src: Any, env: dict[str, Any]) -> list:
    """Resolve a LIST source: a bare list-var name, or a map/filter over another list source. This is
    the higher-order-over-collections layer (RSI 6) — map transforms every element, filter keeps the
    ones a condition selects; fold/len consume the result. Turing-reach over data without exec."""
    if isinstance(src, str):
        v = env.get(src)
        return list(v) if isinstance(v, (list, tuple)) else []
    kind = src[0]
    if kind == "map":
        _, body, inner = src
        return [evaluate(body, {**env, _ELEM: int(x)}) for x in evaluate_list(inner, env)]
    if kind == "filter":
        _, cond, inner = src
        return [int(x) for x in evaluate_list(inner, env) if _eval_cond(cond, {**env, _ELEM: int(x)})]
    return []


def _src_source(src: Any) -> str:
    if isinstance(src, str):
        return src
    kind = src[0]
    if kind == "map":
        return f"map({_ELEM}->{to_source(src[1])}, {_src_source(src[2])})"
    if kind == "filter":
        _, cnd, inner = src
        _, cop, cl, cr = cnd
        return f"filter({to_source(cl)} {cop} {to_source(cr)}, {_src_source(inner)})"
    return str(src)


def to_source(tree: Any) -> str:
    kind = tree[0]
    if kind == "var":
        return str(tree[1])
    if kind == "const":
        return str(tree[1])
    if kind == "len":
        return f"len({_src_source(tree[1])})"
    if kind == "fold":
        _, op, init_t, src = tree
        return f"fold({op!r}, {to_source(init_t)}, {_src_source(src)})"
    if kind == "if":
        _, cond, then_t, else_t = tree
        _, cop, cl, cr = cond
        return f"({to_source(then_t)} if {to_source(cl)} {cop} {to_source(cr)} else {to_source(else_t)})"
    if kind == "let":
        _, name, val_t, body_t = tree
        return f"(let {name} = {to_source(val_t)} in {to_source(body_t)})"
    _, op, left, right = tree
    return f"({to_source(left)} {op} {to_source(right)})"


def _random_cond(vars_: list[str], rng: random.Random) -> Any:
    return ("cmp", rng.choice(list(_CMP)),
            random_tree(vars_, rng, 1, control_flow=False),
            random_tree(vars_, rng, 1, control_flow=False))


def _random_source(list_vars: tuple[str, ...], rng: random.Random) -> Any:
    """A list source: a bare input list, or a map/filter over one (RSI 6). map bodies + filter conds
    are small expressions over the element variable _x — enough for sum-of-squares, sum-of-evens."""
    base = rng.choice(list_vars)
    r = rng.random()
    if r < 0.28:   # map: transform every element (x*x, x+c, x*c, …)
        rhs = ("var", _ELEM) if rng.random() < 0.5 else ("const", rng.randint(1, 3))
        return ("map", ("op", rng.choice(list(_OPS)), ("var", _ELEM), rhs), base)
    if r < 0.46:   # filter: keep elements matching a condition (x%2==0, x>c, …)
        left = (("op", "%", ("var", _ELEM), ("const", 2)) if rng.random() < 0.5 else ("var", _ELEM))
        return ("filter", ("cmp", rng.choice(list(_CMP)), left, ("const", rng.randint(0, 4))), base)
    return base


def random_tree(vars_: list[str], rng: random.Random, depth: int = 2, *, control_flow: bool = False,
                list_vars: tuple[str, ...] = (), library: tuple[Any, ...] = (),
                let_binding: bool = False) -> Any:
    """A random program tree of at most `depth`. Leaves are vars/constants (or len(list) with
    `list_vars`, or a whole reusable sub-program from `library`); `control_flow` allows `if`;
    `list_vars` allows a bounded `fold`; `let_binding` allows a local `let t = value in body` where the
    body may reuse the bound value (RSI layer 7). The library is the COMPOSITIONAL PRIOR: a program
    already solved (sum, len) can be dropped in as one leaf, so composing sum+len is a single grow, not
    a lucky rediscovery of both halves — the No-LLM 'learned building blocks' (DreamCoder-style, sans
    neural recognizer)."""
    if depth <= 0 or (rng.random() < 0.35):
        if library and rng.random() < 0.4:
            return rng.choice(library)                 # reuse a solved sub-program as a leaf
        if list_vars and rng.random() < 0.25:
            return ("len", _random_source(list_vars, rng))
        return ("var", rng.choice(vars_)) if (vars_ and rng.random() < 0.7) else ("const", rng.randint(0, 5))
    sub = lambda v=vars_: random_tree(v, rng, depth - 1, control_flow=control_flow,  # noqa: E731
                                      list_vars=list_vars, library=library, let_binding=let_binding)
    if let_binding and vars_ and rng.random() < 0.18:
        # bind a value to a temp, then build a body that may reuse it (2+ uses = the win over inlining)
        name = f"_t{rng.randint(0, 2)}"
        value = random_tree(vars_, rng, depth - 1, control_flow=control_flow, list_vars=list_vars,
                            library=library)
        body = random_tree(vars_ + [name], rng, depth - 1, control_flow=control_flow,
                           list_vars=list_vars, library=library, let_binding=let_binding)
        return ("let", name, value, body)
    if list_vars and not library and rng.random() < 0.3:
        return ("fold", rng.choice(list(_OPS)), ("const", rng.choice([0, 1])),
                _random_source(list_vars, rng))
    if control_flow and rng.random() < 0.28:
        return ("if", _random_cond(vars_ or list(list_vars), rng), sub(), sub())
    return ("op", rng.choice(list(_OPS)), sub(), sub())


def _subtrees(t: Any):
    """Yield every node in a tree (for finding a repeated subexpression to factor into a let)."""
    if isinstance(t, tuple) and t:
        yield t
        for c in t[1:]:
            if isinstance(c, tuple):
                yield from _subtrees(c)


def _count(t: Any, target: Any) -> int:
    return sum(1 for s in _subtrees(t) if s == target)


def _replace(t: Any, target: Any, repl: Any) -> Any:
    if t == target:
        return repl
    if isinstance(t, tuple) and t:
        return tuple([t[0]] + [_replace(c, target, repl) if isinstance(c, tuple) else c for c in t[1:]])
    return t


def _has_temp(t: Any) -> bool:
    if isinstance(t, tuple) and t:
        if t[0] == "var" and isinstance(t[1], str) and t[1].startswith("_t"):
            return True
        if t[0] == "let":
            return True
        return any(_has_temp(c) for c in t[1:] if isinstance(c, tuple))
    return False


def factor_let(tree: Any, rng: random.Random) -> Any:
    """The let payoff: find a non-trivial subexpression that occurs 2+ times, compute it ONCE under a
    temp name, and replace every occurrence with that name — `(a+b)*(a+b)` → `let t=(a+b) in t*t`. This
    collapses the search for a repeated motif (discover it once, reuse it), and shrinks the program.
    Only factors temp-free arithmetic/list subexpressions so scoping stays sound (value is evaluated in
    the outer env; the name lives only in the rewritten body)."""
    cands = [s for s in set(_subtrees(tree))
             if s[0] in ("op", "if", "fold", "len") and not _has_temp(s) and _count(tree, s) >= 2]
    if not cands:
        return tree
    target = rng.choice(cands)
    name = f"_t{rng.randint(0, 2)}"
    if _count(target, ("var", name)) or _has_temp(tree):     # avoid shadowing an existing temp
        name = f"_t{rng.randint(3, 6)}"
    body = _replace(tree, target, ("var", name))
    return ("let", name, target, body)


def mutate(tree: Any, vars_: list[str], rng: random.Random, *, control_flow: bool = False,
           list_vars: tuple[str, ...] = (), library: tuple[Any, ...] = (),
           let_binding: bool = False) -> Any:
    """One local edit: change an operator/comparison, tweak a constant, swap a leaf's variable/list,
    regrow a small subtree, COMPOSE the whole tree with a library building block, or (with
    `let_binding`) FACTOR a repeated subexpression into a local `let`. The verifier's failing examples
    are the (symbolic) 'gradient' that selection follows — a wrong output nudges the population."""
    kind = tree[0]
    if let_binding and kind != "let":
        if rng.random() < 0.09:
            # self-apply: S -> (S op S). This CREATES the repeated subexpression (e.g. a+b -> (a+b)*
            # (a+b) = a square) that factor_let then names — so x^2-shaped targets become reachable.
            return ("op", rng.choice(list(_OPS)), tree, tree)
        if rng.random() < 0.14:
            factored = factor_let(tree, rng)              # name & reuse a repeated subexpression
            if factored is not tree:
                return factored
    if kind == "let":
        _, name, val_t, body_t = tree
        if rng.random() < 0.5:
            return ("let", name, mutate(val_t, vars_, rng, control_flow=control_flow,
                                        list_vars=list_vars, library=library), body_t)
        return ("let", name, val_t, mutate(body_t, vars_ + [name], rng, control_flow=control_flow,
                                           list_vars=list_vars, library=library, let_binding=let_binding))
    if library and rng.random() < 0.16:
        # compose: (current program) OP (a solved sub-program) — this is the single grow that turns
        # a discovered `sum` into `sum + len`, so composition is reachable, not a lucky rediscovery.
        return ("op", rng.choice(list(_OPS)), tree, rng.choice(library))
    if kind == "fold":
        _, op, init_t, src = tree
        r = rng.random()
        if r < 0.4:
            return ("fold", rng.choice(list(_OPS)), init_t, src)           # flip the fold operator
        if r < 0.7:
            return ("fold", op, mutate(init_t, vars_, rng, list_vars=list_vars), src)  # tweak init
        return ("fold", op, init_t, _random_source(list_vars, rng) if list_vars else src)  # re-roll source
    if kind == "len":
        return ("len", _random_source(list_vars, rng)) if list_vars else tree
    if kind == "if":
        r = rng.random()
        if r < 0.3:   # mutate the condition (operator or a side)
            _, cop, cl, cr = tree[1]
            if rng.random() < 0.5:
                return ("if", ("cmp", rng.choice(list(_CMP)), cl, cr), tree[2], tree[3])
            return ("if", ("cmp", cop, mutate(cl, vars_, rng), cr), tree[2], tree[3])
        if r < 0.65:
            return ("if", tree[1], mutate(tree[2], vars_, rng, control_flow=control_flow), tree[3])
        return ("if", tree[1], tree[2], mutate(tree[3], vars_, rng, control_flow=control_flow))
    if kind == "op":
        r = rng.random()
        if control_flow and r < 0.12:   # grow a conditional in place
            return ("if", _random_cond(vars_, rng), tree[2], tree[3])
        if r < 0.35:
            return ("op", rng.choice(list(_OPS)), tree[2], tree[3])  # flip the operator
        if r < 0.65:
            return ("op", tree[1], mutate(tree[2], vars_, rng, control_flow=control_flow), tree[3])
        return ("op", tree[1], tree[2], mutate(tree[3], vars_, rng, control_flow=control_flow))
    if kind == "const":
        return ("const", max(0, tree[1] + rng.choice([-2, -1, 1, 2])))
    if kind == "var":
        r = rng.random()
        if r < 0.4:
            return ("var", rng.choice(vars_))
        if r < 0.6:
            return random_tree(vars_, rng, 1, control_flow=control_flow,
                               list_vars=list_vars, library=library)        # grow a small subtree
    return tree


def fitness(tree: Any, tests: list[tuple[dict[str, Any], int]]) -> float:
    """THE VERIFIER as fitness — the fraction of input→output examples the program satisfies EXACTLY.
    This is the exact free oracle code gives you and language never does; no gradient is estimated.
    Correctness (solved) is always judged by this."""
    if not tests:
        return 0.0
    ok = sum(1 for env, want in tests if evaluate(tree, env) == want)
    return ok / len(tests)


def graded_fitness(tree: Any, tests: list[tuple[dict[str, Any], int]]) -> float:
    """A SMOOTHED verifier signal for SEARCH (not for correctness). Exact correctness still dominates
    (the integer count of exact hits), but each near-miss earns partial credit for how CLOSE it is
    (1/(1+|got-want|)). This gives selection a continuous landscape over a discontinuous space — the
    symbolic replacement for a gradient: a program that is numerically close climbs toward exact, so
    compositional targets (sum+len, abs) become reachable without any neural core."""
    if not tests:
        return 0.0
    exact, close = 0, 0.0
    for env, want in tests:
        got = evaluate(tree, env)
        if got == want:
            exact += 1
        else:
            close += 1.0 / (1.0 + abs(got - want))
    n = len(tests)
    return exact / n + 0.25 * (close / n)   # exact dominates; near-miss only guides & breaks ties


def evolve(tests: list[tuple[dict[str, Any], int]], vars_: list[str], *, pop: int = 60,
           generations: int = 60, rng_seed: int = 7, control_flow: bool = False,
           list_vars: tuple[str, ...] = (), library: tuple[Any, ...] = (), let_binding: bool = False,
           log=None) -> dict[str, Any]:
    """Gradient-free program search: a population of trees, ranked by the smoothed verifier, elitism +
    mutated offspring, until a candidate passes every example EXACTLY (or the budget ends). No
    differentiable core — the interpreter's free judgment is the only signal. `control_flow` allows
    conditionals; `list_vars` allows bounded folds; `library` supplies solved sub-programs as reusable
    building blocks; `let_binding` allows local variables (compute once, reuse) — the search can factor
    a repeated subexpression into a `let`, collapsing rediscovery of the same motif."""
    rng = random.Random(rng_seed)
    kw = dict(control_flow=control_flow, list_vars=list_vars, library=library, let_binding=let_binding)
    population = [random_tree(vars_, rng, depth=3, **kw) for _ in range(pop)]
    best, best_exact, solved_gen, gen = None, -1.0, None, 0
    for gen in range(1, generations + 1):
        # SEARCH by the smoothed signal (guides toward near-misses); JUDGE correctness by exact.
        scored = sorted(((graded_fitness(t, tests), t) for t in population), key=lambda x: -x[0])
        for _g, t in scored[: max(4, pop // 8)]:   # promote the best EXACT among the graded leaders
            e = fitness(t, tests)
            if e > best_exact:
                best_exact, best = e, t
        if log:
            log(f"[code gen {gen}] exact={best_exact:.2f} graded={scored[0][0]:.3f} -> {to_source(best)}")
        if best_exact >= 1.0:
            solved_gen = gen
            break
        elite = [t for _g, t in scored[: max(2, pop // 6)]]
        population = list(elite)
        while len(population) < pop:
            population.append(mutate(rng.choice(elite), vars_, rng, **kw))
    return {"solved": best_exact >= 1.0, "fitness": round(best_exact, 4),
            "program": to_source(best) if best else None, "tree": best,
            "generation": solved_gen, "generations_run": min(gen, generations)}


def evolve_with_library(problems: list[dict[str, Any]], *, log=None, **kw: Any) -> dict[str, Any]:
    """Self-improving compositional search (owner: keep breaking bricks). Solve a CURRICULUM of
    problems in order; each solved program is added to the LIBRARY, so later, harder problems can
    reuse earlier ones as building blocks. This is the No-LLM 'learned prior': the engine gets better
    at coding by remembering what it already coded — sum + len becomes trivial once sum and len are
    known. Each problem is {tests, vars_, list_vars?, control_flow?}."""
    library: list[Any] = []
    results = []
    for i, p in enumerate(problems):
        r = evolve(p["tests"], p.get("vars_", []), list_vars=p.get("list_vars", ()),
                   control_flow=p.get("control_flow", False), library=tuple(library),
                   pop=p.get("pop", 120), generations=p.get("generations", 200), log=log)
        results.append({"name": p.get("name", f"p{i}"), "solved": r["solved"],
                        "generation": r["generation"], "program": r["program"]})
        if r["solved"] and r["tree"] is not None:
            library.append(r["tree"])       # remember it — a new reusable building block
    return {"solved_count": sum(1 for r in results if r["solved"]), "total": len(problems),
            "library_size": len(library), "results": results}


def synthesize_verified(train_tests: list[tuple[dict[str, int], int]],
                        holdout_tests: list[tuple[dict[str, int], int]], vars_: list[str],
                        **kw: Any) -> dict[str, Any]:
    """The ISOLATION gate for synthesized code (owner + Gemini: the plugin evolves in a sandbox, the
    fortress only accepts a verified result). A program is evolved on TRAIN, then judged on a HELD-OUT
    set it never optimized against — the same frozen-oracle discipline the Critic uses, one level over:
    a program that fits the training examples but FAILS the holdout is OVERFIT and rejected (accepted:
    False). Only a program that also generalizes is offered for the fortress's structural inspection
    (critic_integrity). Nothing here is exec'd; the tree is interpreted, and acceptance is advisory —
    a human still decides whether to mount it."""
    res = evolve(train_tests, vars_, **kw)
    tree = res.get("tree")
    holdout_fit = fitness(tree, holdout_tests) if tree else 0.0
    accepted = bool(res["solved"] and holdout_fit >= 1.0)
    return {**res, "holdout_fitness": round(holdout_fit, 4), "accepted": accepted,
            "verdict": ("generalizes — offer to the fortress inspection gate" if accepted
                        else "overfit or unsolved — rejected at the sandbox boundary")}
