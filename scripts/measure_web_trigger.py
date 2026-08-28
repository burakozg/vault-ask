#!/usr/bin/env python
"""Calibrate `web.thin_distance` — when is the vault's own answer too thin?

    uv run python scripts/measure_web_trigger.py
    uv run python scripts/measure_web_trigger.py --covered q.txt --uncovered u.txt

Prints the distance distribution for questions the vault *does* cover against
questions it does not, and reports whether the configured threshold separates
them. Never searches the web; never calls a generation model.

**Why a distance and not a score.** The first attempt thresholded the fused RRF
score and could not work: RRF is rank-based, so the top hit scores 1/61 =
0.0164 whether it is a perfect match or nonsense. Measured, "what is the
airspeed velocity of an unladen swallow?" scored exactly 0.0164 against this
vault — indistinguishable from a well-covered question. The fused score
carries no relevance information at all. Vector distance does.

Needs a populated index and a reachable Ollama host (the query has to be
embedded), so it is a script rather than a test — same reasoning as
`measure_graph_expansion.py`.

Re-run after a large corpus change or an embedding model swap: the numbers are
specific to bge-m3 and to what this vault happens to contain.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from vault_ask.ask import _should_search_web, retrieve
from vault_ask.config import load_settings
from vault_ask.db import connect

#: Subjects this vault demonstrably holds material on.
COVERED = [
    "What have I read about AI agents?",
    "What do my notes say about prompt injection?",
    "What is in my notes about Entra ID?",
    "What have I collected about zero trust?",
    "What does my reading say about AI coding assistants?",
    "What do my notes say about ransomware incidents?",
]

#: Deliberately outside the vault's subject matter, and deliberately ordinary
#: questions rather than gibberish — a trigger that only fires on nonsense is
#: not useful. These are things a person might genuinely ask and the vault
#: genuinely cannot answer.
UNCOVERED = [
    "How do I make a sourdough starter from scratch?",
    "What is the offside rule in football?",
    "Who won the 1974 FIFA World Cup?",
    "What is the airspeed velocity of an unladen swallow?",
    "How do I repot a fiddle leaf fig?",
    "What is the capital of Mongolia?",
]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--covered", type=Path, help="file of questions the vault covers")
    parser.add_argument("--uncovered", type=Path, help="file of questions it does not")
    args = parser.parse_args()

    cfg = load_settings(args.config)
    if not cfg.models.embedding_base_url:
        print(
            "models.embedding_base_url is unset — without the vector arm there is no "
            "distance to measure. Set VAULTASK_MODELS__EMBEDDING_BASE_URL "
            "(see OLLAMA-SETUP.md).",
            file=sys.stderr,
        )
        return 2

    def _load(path: Path | None, fallback: list[str]) -> list[str]:
        if path is None:
            return fallback
        return [q.strip() for q in path.read_text().splitlines() if q.strip()]

    covered = _load(args.covered, COVERED)
    uncovered = _load(args.uncovered, UNCOVERED)

    conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
    groups: dict[str, list[tuple[str, float | None, bool]]] = {}
    try:
        for label, questions in (("COVERED", covered), ("NOT COVERED", uncovered)):
            rows = []
            for question in questions:
                # allow_web=True is the mode the trigger runs in, and the mode
                # that restricts retrieval to `open` chunks. Measuring in any
                # other mode would calibrate against a corpus the feature can
                # never actually see.
                hits = await retrieve(conn, cfg, question, allow_web=True)
                distances = [h.distance for h in hits if h.distance is not None]
                best = min(distances) if distances else None
                rows.append((question, best, _should_search_web(cfg, hits, allow_web=True)))
            groups[label] = rows
    finally:
        conn.close()

    width = max(len(q) for q in covered + uncovered)
    for label, rows in groups.items():
        print(f"\n--- {label} ---")
        for question, best, fires in rows:
            shown = "  n/a " if best is None else f"{best:6.4f}"
            print(f"  {question.ljust(width)}  dist={shown}  searches_web={fires}")

    def _stats(label: str) -> tuple[float, float] | None:
        vals = [b for _, b, _ in groups[label] if b is not None]
        return (min(vals), max(vals)) if vals else None

    cov, unc = _stats("COVERED"), _stats("NOT COVERED")
    print(f"\nthreshold in use: web.thin_distance = {cfg.web.thin_distance}")
    if cov:
        print(f"  covered     : {cov[0]:.4f} to {cov[1]:.4f}")
    if unc:
        print(f"  not covered : {unc[0]:.4f} to {unc[1]:.4f}")

    if cov and unc:
        if cov[1] < unc[0]:
            midpoint = (cov[1] + unc[0]) / 2
            print(f"  ✓ cleanly separated. Gap {cov[1]:.4f}-{unc[0]:.4f}; midpoint {midpoint:.4f}.")
            if not (cov[1] < cfg.web.thin_distance < unc[0]):
                print(f"  ✗ but thin_distance={cfg.web.thin_distance} is OUTSIDE that gap — "
                      f"set it to about {midpoint:.4f}.")
        else:
            print(f"  ✗ OVERLAPPING ({unc[0]:.4f} < {cov[1]:.4f}). No single threshold separates "
                  "these sets — the trigger will misfire either way, and the question set or the "
                  "signal needs rethinking rather than the number nudging.")

    wrong = [q for q, _, f in groups["COVERED"] if f] + [
        q for q, _, f in groups["NOT COVERED"] if not f
    ]
    print(f"\nmisclassified at the current threshold: {len(wrong)}")
    for q in wrong:
        print(f"  - {q}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
