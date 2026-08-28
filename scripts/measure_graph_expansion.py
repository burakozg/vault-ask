#!/usr/bin/env python
"""A/B one-hop graph expansion against a real index, and report what it changes.

    uv run python scripts/measure_graph_expansion.py
    uv run python scripts/measure_graph_expansion.py --questions my-questions.txt

Deliberately a script, not a test — like podcast-digest's
`scripts/check-enclosure-chains.py`, it needs things CI does not have: a
populated `index.sqlite` (2,254 docs here) and a reachable Ollama host to embed
each query. Follows `clippings-topics`' `--calibrate` precedent: run the real
pipeline, print a table, write nothing.

**Why this exists.** README "Build order" step 5 has said graph expansion is
"not yet measured against real questions (the README's own bar for keeping
it)" since it landed. That was a statement about missing *instrumentation*, not
missing effort: `SearchHit` carried no provenance, so nothing downstream could
tell an expanded chunk from a direct hit, and `graph_max_siblings=0` was not an
off switch (topic notes inject independently of the sibling query). Both are
fixed — `SearchHit.source` and `retrieval.graph_enabled` — so the comparison is
now possible.

**What to look for.** Expansion reaches only docs under a topic note: on this
vault, 65 topic notes covering 81 docs, ~3.6% of the corpus. A question set
sampled at random will mostly not exercise it at all, so the fixture below
deliberately mixes graph-reachable topics with unreachable ones — a fair test
has to include the cases where expansion should do nothing.

The specific claim under test is README "Retrieval"'s: that the 0.7 discount
means an expanded chunk "wins only when genuinely competitive". Arithmetic
says otherwise. With `_RRF_K = 60`, a hit ranked 1 in *both* arms scores
`2/61 = 0.0328`; discounted, the chunk it pulls in scores `0.0230` — which
outranks a rank-1 *single-arm* hit at `1/61 = 0.0164`. So an expanded chunk,
never scored against the query at all, can displace a genuine top hit. The
`displaced` column below is what quantifies how often that actually happens.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from vault_ask.ask import retrieve
from vault_ask.config import Settings, load_settings
from vault_ask.db import connect

#: Mixed on purpose — see the module docstring. Roughly half aimed at subjects
#: with topic notes in `99 topics/` (where expansion can fire), half at
#: material that is indexed but ungrouped (where it must not).
DEFAULT_QUESTIONS = [
    # Likely graph-reachable (topic notes exist for these).
    "What have I read about AI agents?",
    "What do my notes say about LLM evaluation?",
    "What are the main themes in my AI security reading?",
    "What have I collected about prompt injection?",
    "What does my vault say about RAG architectures?",
    "What have I read about Claude and Anthropic?",
    "What are the recurring ideas about agentic engineering?",
    "What do my notes cover on model context protocol?",
    "What have I saved about fine-tuning?",
    "What does my reading say about AI coding assistants?",
    # Likely not graph-reachable (indexed, ungrouped).
    "What happened with Salesforce and AI growth?",
    "What do my notes say about ransomware incidents?",
    "What have I read about CISO priorities?",
    "What is in my notes about Entra ID?",
    "What have I read about supply chain attacks?",
    "What do my podcast notes say about threat intelligence?",
    "What have I collected about zero trust?",
    "What does my vault say about incident response?",
    "What have I read about cloud misconfiguration?",
    "What do my notes say about phishing campaigns?",
]


def _with_graph(cfg: Settings, enabled: bool) -> Settings:
    return cfg.model_copy(
        update={"retrieval": cfg.retrieval.model_copy(update={"graph_enabled": enabled})}
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--questions", type=Path, default=None, help="file with one question per line"
    )
    parser.add_argument("--verbose", action="store_true", help="show each displaced hit")
    args = parser.parse_args()

    cfg = load_settings(args.config)
    questions = (
        [q.strip() for q in args.questions.read_text().splitlines() if q.strip()]
        if args.questions
        else DEFAULT_QUESTIONS
    )

    if not cfg.models.embedding_base_url:
        print(
            "models.embedding_base_url is unset — this would measure FTS-only "
            "retrieval and tell you nothing about expansion. Set "
            "VAULTASK_MODELS__EMBEDDING_BASE_URL (see OLLAMA-SETUP.md).",
            file=sys.stderr,
        )
        return 2

    conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
    on, off = _with_graph(cfg, True), _with_graph(cfg, False)

    total_expanded = 0
    total_displaced = 0
    fired_on = 0
    rows: list[tuple[str, int, int, int]] = []

    try:
        for question in questions:
            hits_on = await retrieve(conn, on, question, allow_web=False)
            hits_off = await retrieve(conn, off, question, allow_web=False)

            expanded = [h for h in hits_on if h.source == "graph"]
            # What expansion pushed out of the final top-k: chunks that were in
            # the baseline answer and are not in the expanded one.
            baseline_ids = [h.chunk_id for h in hits_off]
            kept_ids = {h.chunk_id for h in hits_on}
            displaced = [cid for cid in baseline_ids if cid not in kept_ids]

            rows.append((question, len(hits_on), len(expanded), len(displaced)))
            total_expanded += len(expanded)
            total_displaced += len(displaced)
            fired_on += 1 if expanded else 0

            if args.verbose and expanded:
                print(f"\n  {question}")
                for h in expanded:
                    print(f"    + graph  {h.score:.6f}  {h.path}")
                for cid in displaced:
                    lost = next(h for h in hits_off if h.chunk_id == cid)
                    print(f"    - lost   {lost.score:.6f}  [{lost.source}] {lost.path}")
    finally:
        conn.close()

    width = max(len(q) for q in questions)
    print(f"\n{'question'.ljust(width)}  hits  graph  displaced")
    print("-" * (width + 20))
    for question, n_hits, n_exp, n_disp in rows:
        print(f"{question.ljust(width)}  {n_hits:>4}  {n_exp:>5}  {n_disp:>9}")

    n = len(questions)
    print("-" * (width + 20))
    print(f"{'TOTAL'.ljust(width)}  {'':>4}  {total_expanded:>5}  {total_displaced:>9}")
    print()
    print(f"questions                        : {n}")
    print(f"questions where expansion fired  : {fired_on} ({fired_on / n:.0%})")
    print(f"expanded chunks in final answers : {total_expanded}")
    print(f"direct hits they displaced       : {total_displaced}")
    print()
    print(
        "A high `displaced` relative to `graph` means expansion is not additive: it is\n"
        "evicting chunks that were actually scored against the question in favour of\n"
        "chunks that never were. That is the README claim under test."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
