# -*- coding: utf-8 -*-
"""ATANOR VibeCode — a No-LLM, verifier-guided program-synthesis engine that learns to code by
trial-and-error against a free exact oracle (the interpreter), with NO gradient and NO neural core.

Three layers, each a standalone brick:
  code_evolver     — gradient-free program search: a population of expression TREES evolves, judged
                     only by how many input→output examples it satisfies (the verifier IS the fitness).
  auto_curriculum  — the engine invents its own problems (composing its verified library), solves and
                     generalizes them, dedups by BEHAVIOR (capability = the function, keep the smallest
                     program), and self-paces difficulty (competence vs novelty). No human writes targets.
  abstraction      — primitive invention by anti-unification (DreamCoder-style, no neural recognizer):
                     recurring motifs across the library are generalized into new named primitives, so
                     even the axiom set expands on its own.

SAFETY: every genome is a whitelisted tree (vars, small int constants, + - * // %, comparisons,
bounded fold/len/map/filter) INTERPRETED directly — never compiled or exec'd. A candidate can only
ever do arithmetic on the given numbers. Acceptance passes a held-out generalization gate.
"""
from __future__ import annotations

from atanor_vibecode.abstraction import anti_unify, instantiate, mine
from atanor_vibecode.auto_curriculum import autonomous_round, new_state, run
from atanor_vibecode.code_evolver import (
    evaluate,
    evolve,
    evolve_with_library,
    fitness,
    graded_fitness,
    synthesize_verified,
    to_source,
)

__version__ = "0.1.0"

__all__ = [
    # synthesis
    "evaluate", "evolve", "evolve_with_library", "fitness", "graded_fitness",
    "synthesize_verified", "to_source",
    # self-curriculum
    "run", "autonomous_round", "new_state",
    # primitive invention
    "mine", "anti_unify", "instantiate",
]
