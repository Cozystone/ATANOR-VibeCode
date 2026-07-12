# -*- coding: utf-8 -*-
"""Primitive invention by anti-unification (owner 2026-07-13: advance without a human adding
primitives). The self-curriculum grows COMPOSITIONAL DEPTH over a fixed axiom set; this is the next
brick — the engine INVENTS its own new primitives, so even the axiom set expands on its own.

THE MECHANISM (DreamCoder's abstraction/compression, with NO neural recognizer).
After solving programs, look for sub-programs that recur across the library differing only in their
leaves — e.g. `a + a*a` and `b + b*b`. Anti-unify them: keep the shared skeleton, replace the points
where they differ with HOLES, yielding a parameterized template `λx. x + x*x` (the "square-plus-self"
motif). That template is a new named primitive; instantiating it (binding the holes to concrete
sub-programs) produces an ordinary tree the existing safe interpreter already runs — so the invented
primitive costs NO change to the interpreter kernel and can only ever do whitelisted arithmetic.

An abstraction earns its name by COMPRESSION: if factoring the motif out of the library saves nodes
(it recurs, and it's non-trivial), it's worth keeping. This is the honest signal — a primitive that
doesn't compress anything is just noise, and is rejected.

SAFETY: templates are the same whitelisted tree grammar as code_evolver, with `("hole", i)` marking
a parameter slot; instantiation substitutes concrete subtrees, never code. Nothing is exec'd.
"""
from __future__ import annotations

from typing import Any, Optional

from atanor_vibecode.code_evolver import to_source

_HOLE = "hole"


def _is_node(t: Any) -> bool:
    return isinstance(t, (tuple, list)) and len(t) > 0


def _tag(t: Any) -> Any:
    return t[0] if _is_node(t) else None


def size(t: Any) -> int:
    if not _is_node(t):
        return 1
    return 1 + sum(size(c) for c in t[1:] if _is_node(c))


def _hole_indices(t: Any, acc: set) -> set:
    if not _is_node(t):
        return acc
    if _tag(t) == _HOLE:
        acc.add(t[1])
        return acc
    for c in t[1:]:
        if _is_node(c):
            _hole_indices(c, acc)
    return acc


def holes_in(t: Any) -> int:
    """The template's ARITY — the number of DISTINCT parameters (a repeated hole counts once)."""
    return len(_hole_indices(t, set()))


def _hole_occ(t: Any) -> int:
    """How many hole NODES appear (a repeated parameter counts each time) — for body-size math."""
    if not _is_node(t):
        return 0
    if _tag(t) == _HOLE:
        return 1
    return sum(_hole_occ(c) for c in t[1:] if _is_node(c))


def anti_unify(t1: Any, t2: Any, _state: Optional[dict] = None) -> Any:
    """Generalize two trees to their least-general common template. Where they agree, structure is
    kept; where they differ, a `("hole", i)` is introduced — the parameter of the new primitive. A
    memo table gives the SAME hole to identical disagreement pairs, so a repeated variable is captured
    as one parameter: `a + a*a` vs `b + b*b` → `x0 + x0*x0` (λx. x + x*x), not three holes."""
    if _state is None:
        _state = {"ctr": [0], "memo": {}}

    def hole_for(a: Any, b: Any) -> Any:
        key = (repr(a), repr(b))
        memo = _state["memo"]
        if key not in memo:
            memo[key] = (_HOLE, _state["ctr"][0])
            _state["ctr"][0] += 1
        return memo[key]

    if t1 == t2:
        return t1
    if not (_is_node(t1) and _is_node(t2)):
        return hole_for(t1, t2)
    # same structural shape? same tag, same arity, and for operator-bearing nodes the SAME operator
    if _tag(t1) != _tag(t2) or len(t1) != len(t2):
        return hole_for(t1, t2)
    tag = _tag(t1)
    if tag in ("op", "cmp", "fold") and t1[1] != t2[1]:
        return hole_for(t1, t2)                            # different operator → generalize whole node
    if tag == "var" or tag == "const":
        return hole_for(t1, t2)                            # differing leaves → a parameter
    # recurse element-wise, holding the tag (and operator slot) fixed
    out = [tag]
    fixed = 2 if tag in ("op", "cmp", "fold") else 1
    for i in range(1, len(t1)):
        if i < fixed:
            out.append(t1[i])                              # keep operator symbol / fold operator
        else:
            out.append(anti_unify(t1[i], t2[i], _state))
    return tuple(out)


def _renumber(t: Any, mapping: dict) -> Any:
    """Make hole indices contiguous 0..k-1 in first-appearance order (a canonical template form)."""
    if not _is_node(t):
        return t
    if _tag(t) == _HOLE:
        if t[1] not in mapping:
            mapping[t[1]] = len(mapping)
        return (_HOLE, mapping[t[1]])
    return tuple([t[0]] + [_renumber(c, mapping) if _is_node(c) else c for c in t[1:]])


def canonical(template: Any) -> Any:
    return _renumber(template, {})


def match(template: Any, tree: Any, binds: Optional[dict] = None) -> Optional[dict]:
    """Is `tree` an instance of `template`? Returns hole→subtree bindings if so (consistent reuse of a
    hole must bind the same subtree), else None."""
    if binds is None:
        binds = {}
    if _is_node(template) and _tag(template) == _HOLE:
        i = template[1]
        if i in binds:
            return binds if binds[i] == tree else None
        binds[i] = tree
        return binds
    if not (_is_node(template) and _is_node(tree)):
        return binds if template == tree else None
    if _tag(template) != _tag(tree) or len(template) != len(tree):
        return None
    tag = _tag(template)
    fixed = 2 if tag in ("op", "cmp", "fold") else 1
    for i in range(1, len(template)):
        if i < fixed:
            if template[i] != tree[i]:
                return None
        else:
            if match(template[i], tree[i], binds) is None:
                return None
    return binds


def instantiate(template: Any, args: list) -> Any:
    """Bind the template's holes to concrete subtrees, producing an ordinary (hole-free) tree."""
    if not _is_node(template):
        return template
    if _tag(template) == _HOLE:
        return args[template[1]]
    return tuple([template[0]] + [instantiate(c, args) if _is_node(c) else c for c in template[1:]])


def _all_subtrees(t: Any):
    if not _is_node(t):
        return
    yield t
    for c in t[1:]:
        if _is_node(c):
            yield from _all_subtrees(c)


def compression_gain(library: list, template: Any) -> int:
    """Nodes saved by naming this abstraction: for every library subtree that matches the template,
    each occurrence collapses to a single application, so the saving is (template_body_size - 1) per
    occurrence, counted only when it recurs (≥2). A template that compresses nothing scores 0."""
    body = size(template) - _hole_occ(template)            # the non-hole structure that gets shared
    ctmpl = canonical(template)
    occ = sum(1 for lib in library for st in _all_subtrees(lib) if match(ctmpl, st, {}) is not None)
    if occ < 2 or body < 2:
        return 0
    return (body - 1) * occ


def _wf_source(s: Any) -> bool:
    """A list-source is well-formed if it is a bare list var, or a map/filter over one — never a hole
    (a hole binds to an int-expression, which is not a list)."""
    if isinstance(s, str):
        return True
    if not _is_node(s):
        return False
    tag = _tag(s)
    if tag == "map":
        return _well_formed(s[1]) and _wf_source(s[2])
    if tag == "filter":
        cond = s[1]
        return _is_node(cond) and _tag(cond) == "cmp" and _well_formed(cond) and _wf_source(s[2])
    return False                                            # hole or anything else → not a valid source


def _well_formed(t: Any) -> bool:
    """A template is usable only if every hole sits in a VALUE position — never as an if/filter
    condition (must stay a cmp) or a fold/len list-source (must stay a list). Otherwise instantiating
    it would produce a tree the safe interpreter can't render or evaluate."""
    if not _is_node(t):
        return True
    tag = _tag(t)
    if tag == _HOLE:
        return True
    if tag == "op":
        return _well_formed(t[2]) and _well_formed(t[3])
    if tag == "if":
        cond = t[1]
        if not (_is_node(cond) and _tag(cond) == "cmp"):
            return False
        return _well_formed(cond) and _well_formed(t[2]) and _well_formed(t[3])
    if tag == "cmp":
        return _well_formed(t[2]) and _well_formed(t[3])
    if tag == "fold":
        return _well_formed(t[2]) and _wf_source(t[3])
    if tag == "len":
        return _wf_source(t[1])
    return False                                            # map/filter/etc. cannot be a template ROOT


def _body_has_var(t: Any) -> bool:
    """Does the template keep a bare VARIABLE in its (non-hole) body? A pinned variable means the
    abstraction failed to parameterize an incidental variable two programs happened to share — low
    value. A kept CONSTANT (e.g. the 2 in `2*x`) is real semantics and is fine."""
    if not _is_node(t):
        return False
    if _tag(t) == _HOLE:
        return False
    if _tag(t) == "var":
        return True
    return any(_body_has_var(c) for c in t[1:] if _is_node(c))


def mine(library: list, *, top_k: int = 6, min_gain: int = 2, max_pool: int = 80) -> list[dict]:
    """Invent primitives: pool the SUBTREES of every library program (so recurring INTERNAL motifs
    surface, not just whole-program roots) and anti-unify pairs of them into parameterized templates.
    Keep the templates that COMPRESS the library (recur, non-trivial, 1–2 holes, no pinned variable in
    the body), ranked by compression gain. Each result is a reusable primitive the generator/search can
    instantiate — a new named building block the engine invented for itself."""
    pool: list = []                                        # distinct VALUE-producing subtrees
    for lib in library:
        for st in _all_subtrees(lib):
            if (_tag(st) in ("op", "if", "fold", "len") and 3 <= size(st) <= 10
                    and not _hole_occ(st) and st not in pool):
                pool.append(st)
        if len(pool) >= max_pool:
            break

    seen: dict[str, dict] = {}
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            tmpl = canonical(anti_unify(pool[i], pool[j]))
            h = holes_in(tmpl)                             # arity (distinct parameters)
            if not (1 <= h <= 2) or size(tmpl) - _hole_occ(tmpl) < 2 or _body_has_var(tmpl):
                continue
            if not _well_formed(tmpl):                     # holes only in value positions, valid root
                continue
            gain = compression_gain(library, tmpl)
            if gain < min_gain:
                continue
            key = _template_source(tmpl)
            if key not in seen or gain > seen[key]["gain"]:
                seen[key] = {"template": tmpl, "holes": h, "arity": h, "gain": gain, "source": key}
    return sorted(seen.values(), key=lambda d: -d["gain"])[:top_k]


def _template_source(tmpl: Any) -> str:
    """Human-readable λ form of a template, e.g. 'λ(x0,x1). (x0 + (x1 * x1))'."""
    def render(t: Any) -> Any:
        if _is_node(t) and _tag(t) == _HOLE:
            return ("var", f"x{t[1]}")                     # show holes as parameters
        if not _is_node(t):
            return t
        return tuple([t[0]] + [render(c) if _is_node(c) else c for c in t[1:]])
    h = holes_in(tmpl)
    params = ",".join(f"x{i}" for i in range(h))
    return f"λ({params}). {to_source(render(tmpl))}"
