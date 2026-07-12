# -*- coding: utf-8 -*-
"""Code self-curriculum runner (owner 2026-07-13: "코드 파인튜닝 없이도 진전이 꾸준해야 해").

Runs the autonomous code curriculum so the engine's coding capability advances WITHOUT a human
adding primitives or writing targets. Each round the engine invents its own problems (composing its
verified library), solves + generalizes them, dedups by behavior (keeps the smallest program per
FUNCTION, rejects trivial/bloated ones), and the controller self-paces difficulty (climb on
mastery+saturation, ease off when a tier is too hard). Progress accrues in a journal the owner can
watch — no code editing between advances.

Two modes:
  python scripts/code_autocurriculum.py --once 8     # run 8 rounds now, print + exit (on-demand)
  python scripts/code_autocurriculum.py              # bounded daemon: a burst every INTERVAL sec

OPT-IN by design. This is NOT auto-started by the watchdog (the diet-flood lesson: no new always-on
CPU burner without ops control). Singleton (port lock 18793). Offline-only: it writes just
runtime/evolution/curriculum_state.json + curriculum_journal.jsonl; it never touches the stores,
answer packs, or the engine process, so it can never regress P0.
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STATE = REPO / "runtime" / "evolution" / "curriculum_state.json"
JOURNAL = REPO / "runtime" / "evolution" / "curriculum_journal.jsonl"
LOG = REPO / "runtime" / "evolution" / "curriculum_daemon.log"
INTERVAL = int(os.getenv("ATANOR_CURR_INTERVAL_SEC", "1800") or 1800)
ROUNDS = int(os.getenv("ATANOR_CURR_ROUNDS", "4") or 4)          # rounds per burst
PROBLEMS = int(os.getenv("ATANOR_CURR_PROBLEMS", "6") or 6)      # generated problems per round


def _log(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def one_burst() -> dict:
    from atanor_vibecode.auto_curriculum import run

    before = _distinct()
    out = run(rounds=ROUNDS, state_path=STATE, journal_path=JOURNAL, problems=PROBLEMS,
              log=lambda m: _log(f"  {m}"))
    grew = out["distinct_solved"] - before
    _log(f"burst done: tier {out['tier']} | distinct functions {out['distinct_solved']} "
         f"(+{grew} this burst) | frontier {out['frontier']}")
    return out


def _distinct() -> int:
    try:
        from atanor_vibecode.auto_curriculum import load_state
        s = load_state(STATE)
        return sum(len(v) for v in s["sigs"].values())
    except Exception:
        return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--once":
        n = int(argv[1]) if len(argv) > 1 else ROUNDS
        from atanor_vibecode.auto_curriculum import run
        out = run(rounds=n, state_path=STATE, journal_path=JOURNAL, problems=PROBLEMS,
                  log=lambda m: print(m, flush=True))
        print(f"\nFINAL: tier {out['tier']} | distinct functions {out['distinct_solved']}")
        for fam, progs in out["libraries"].items():
            print(f"[{fam}] {len(progs)}: " + " | ".join(sorted(progs, key=len)[:6])
                  + (" …" if len(progs) > 6 else ""))
        return 0

    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 18793))
        lock.listen(1)
    except OSError:
        print("another curriculum runner is already running; exiting")
        return 0
    _log(f"code curriculum daemon up — a burst of {ROUNDS} rounds every {INTERVAL}s "
         f"({PROBLEMS} problems/round)")
    while True:
        try:
            one_burst()
        except Exception as exc:
            _log(f"burst failed: {type(exc).__name__}: {exc}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
