# ATANOR VibeCode

A **No-LLM, verifier-guided program-synthesis engine**. It learns to write programs by
trial-and-error against a free exact oracle — the interpreter — with **no gradient and no neural
core**. Extracted from ATANOR as a standalone, pluggable brick.

## Why this exists

The common claim is that because code is discontinuous (one typo → Fail), you can't learn it by
counting frequencies like language; you need a differentiable neural core to compute which direction
to fix. That is **overstated**. The thing language lacks and code *has* is a free, exact **verifier**:
the interpreter/tests return crisp ground truth (pass/fail) — and that replaces the gradient.
Verifier-guided search over programs (genetic programming, MCTS, CEGIS, type-directed enumeration) is
a decades-old paradigm that learns programs with no gradient. This engine is the smallest honest
demonstration, taken far.

**The real axis** for "can it code X?" is not symbolic-vs-neural — it is **"does a verifier exist?"**
Checkable (tests / spec / compiles / a failing bug that should pass) → this engine handles it, and
because every output is verified, *safely*. Uncheckable (aesthetic, fuzzy) → no method is reliable;
an LLM just guesses a prettier prior. Most valuable engineering coding is checkable — that is this
engine's territory.

## The three layers

| Module | What it does |
|---|---|
| `code_evolver` | Gradient-free program search. A population of expression **trees** evolves, ranked only by the fraction of input→output examples it satisfies (the verifier **is** the fitness). Discovers `a+b`, `max(a,b)`, `sum(xs)=fold(+,0,xs)`, sum-of-evens, etc. from examples alone. Graded fitness + a self-improving library crack composition (`sum+len`). |
| `auto_curriculum` | The engine invents its **own** problems by composing its verified library, solves + generalizes them (held-out gate), dedups by **behavior** (capability = the function computed; keep the smallest program; reject trivial/bloated), and **self-paces** difficulty via two signals — *competence* (does it solve its own targets) and *novelty* (is it still learning). No human writes targets or sets the schedule. |
| `abstraction` | **Primitive invention** by anti-unification (DreamCoder-style, no neural recognizer). Recurring motifs across the library — e.g. `a+a*a` and `b+b*b` — are generalized into new named primitives — `λx. x + x*x` — that earn their name by **compression**. So even the axiom set expands on its own. |

## Safety

Every genome is a **whitelisted tree** (variables, small integer constants, `+ - * // %`, comparisons,
bounded `fold`/`len`/`map`/`filter`) that is **interpreted directly — never compiled or `exec`'d**. A
candidate can only ever do arithmetic on the numbers it is given. Every acceptance passes a **held-out
generalization gate** (train in a sandbox, judge on never-optimized examples; overfit is rejected).

## Quickstart

```bash
pip install -e .
pytest                                   # the full suite, import-mode=importlib

# run the autonomous self-curriculum for 8 rounds and watch capability accrue
python scripts/code_autocurriculum.py --once 8
```

```python
from atanor_vibecode import evolve, run

# discover addition from examples alone (no gradient)
out = evolve([({"a": a, "b": b}, a + b) for a in range(6) for b in range(6)], ["a", "b"])
print(out["program"], out["solved"])     # -> "(a + b)"  True

# let the engine advance on its own (self-curriculum + primitive invention)
state = run(rounds=8)
print(state["frontier"])                 # distinct functions, invented primitives, tier
```

## Honest scope

This accumulates **compositional depth** over a fixed axiom set and **invents new primitives** from
recurring motifs — genuine, measurable capability growth. It does **not** conjure new mathematics from
nothing. Current primitive invention covers pure-value primitives (arithmetic + list reductions);
conditional/list *structure* abstraction, and local variable binding (`let`), are the next bricks.

## License

Proprietary — © Cozystone / ATANOR. All rights reserved.
