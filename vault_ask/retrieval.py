"""Retrieval: FTS5 keyword search, vector search, reciprocal rank fusion
between them, and one-hop graph expansion on top (README "Retrieval"). No
rerank yet — that is still a later step; this module's shape (a ranked list
of `SearchHit`, filterable by sensitivity) is what it plugs into.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import struct
from dataclasses import dataclass, replace

log = logging.getLogger("vault_ask.retrieval")


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    path: str
    title: str
    heading_path: str
    prelude: str
    body: str
    sensitivity: str
    #: Whatever the source ranker produced — bm25 from FTS5, distance from
    #: vec0, or a fused RRF score. Never comparable across sources; see
    #: fuse_rrf, which combines by *rank*, not by this value.
    score: float
    #: Which retrieval arm produced this hit: "vector", "fts", "vector+fts"
    #: (found by both, fused), or "graph" (reached by expansion, never scored
    #: against the query at all — its score is inherited from the hit that
    #: reached it).
    #:
    #: Exists so graph expansion can be *measured* rather than argued about.
    #: fuse_rrf used to discard origin entirely, so by the time an answer was
    #: assembled nothing in the process could tell an expanded chunk from a
    #: direct hit — which made the README's "not yet measured" note a
    #: statement about missing instrumentation, not missing effort.
    source: str = ""
    #: Raw vector distance to the query, when this chunk was found by vector
    #: search. Preserved *through* fusion (dataclasses.replace keeps it) for
    #: one reason: the fused RRF score carries no relevance information at all
    #: — rank 1 scores 1/61 whether the match is perfect or nonsense — so it
    #: cannot answer "does the vault actually cover this?". Distance can.
    #: Read by `vault_ask.ask._should_search_web`.
    distance: float | None = None


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def build_fts_query(text: str) -> str:
    """Free text -> an FTS5 query that will not raise a syntax error.

    Individually-quoted terms joined by OR: quoting strips FTS5's operator
    meaning from things like a lone hyphen or the word "and", and OR (rather
    than the implicit AND between bareword terms) means a question does not
    need every one of its words present in a chunk to surface it at all.
    """
    tokens = _WORD_RE.findall(text)
    return " OR ".join(f'"{t}"' for t in tokens)


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int,
    sensitivity: str | None = None,
) -> list[SearchHit]:
    """Keyword search over the corpus. ``sensitivity`` narrows to that value
    when given — the caller's job (see the allow_web -> 'open' rule in
    README "Corpus and sensitivity"), not this function's to decide.
    """
    fts_query = build_fts_query(query)
    if not fts_query:
        return []

    rows = conn.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.heading_path, c.prelude, c.body, c.sensitivity,
               d.path, d.title, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE chunks_fts MATCH :query
          AND (:sensitivity IS NULL OR c.sensitivity = :sensitivity)
        ORDER BY bm25(chunks_fts)
        LIMIT :top_k
        """,
        {"query": fts_query, "sensitivity": sensitivity, "top_k": top_k},
    ).fetchall()

    return [
        SearchHit(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            path=row["path"],
            title=row["title"] or row["path"],
            heading_path=row["heading_path"] or "",
            prelude=row["prelude"] or "",
            body=row["body"] or "",
            sensitivity=row["sensitivity"],
            score=row["score"],
            source="fts",
        )
        for row in rows
    ]


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


#: How much wider to re-ask vec0 when the sensitivity filter ate the results.
#: 4x rather than +N because the shortfall is proportional: if personal chunks
#: are ~half the neighbourhood, doubling barely helps, and each retry costs a
#: full KNN scan.
_VEC_WIDEN_FACTOR = 4


def search_vector(
    conn: sqlite3.Connection,
    query_vector: list[float],
    *,
    top_k: int,
    sensitivity: str | None = None,
) -> list[SearchHit]:
    """Nearest-neighbour search over `chunks_vec`. Same sensitivity contract as
    search_fts. Raises whatever sqlite3/sqlite-vec raises on a dimension
    mismatch (e.g. the index was built with a different embedding model) —
    callers degrade to FTS-only on failure rather than this function guessing
    at a fallback (see vault_ask.ask).

    **The sensitivity predicate here is a post-filter, unlike search_fts.**
    `chunks_vec` is `vec0(chunk_id, embedding)` — it carries no `sensitivity`
    column, so vec0 cannot see the predicate: `k = :top_k` binds first and
    `c.sensitivity` is evaluated afterwards, on the joined rows vec0 already
    chose. A plain `k = top_k` therefore returns *fewer* than top_k permitted
    chunks whenever personal ones sit nearer the query — measured on
    sqlite-vec 0.1.9 with 10 personal chunks nearer than 3 open ones,
    `top_k=5, sensitivity='open'` returned **zero** rows while open matches
    existed.

    Nothing is leaked by that (the filter does apply), but the vector arm goes
    silently short, and RRF then fuses a full-width FTS list against a stunted
    vector list — tilting every answer toward keyword matching exactly when the
    question is topically near personal material. So: widen k and re-ask until
    top_k permitted rows are found or vec0 runs out of corpus. The alternative,
    giving vec0 a `sensitivity` metadata column, is a schema change that costs
    a full re-embed of every chunk.
    """
    filtering = sensitivity is not None
    limit = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] if filtering else top_k
    k = top_k
    rows: list[sqlite3.Row] = []

    while True:
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.heading_path, c.prelude, c.body, c.sensitivity,
                   d.path, d.title, v.distance AS score
            FROM chunks_vec v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            JOIN docs d ON d.doc_id = c.doc_id
            WHERE v.embedding MATCH :query_vector
              AND k = :k
              AND (:sensitivity IS NULL OR c.sensitivity = :sensitivity)
            ORDER BY v.distance
            """,
            {
                "query_vector": _serialize(query_vector),
                "k": k,
                "sensitivity": sensitivity,
            },
        ).fetchall()

        # Enough permitted rows, or vec0 has already been asked for the whole
        # corpus and cannot return more however wide we go.
        if not filtering or len(rows) >= top_k or k >= limit:
            break
        k = min(k * _VEC_WIDEN_FACTOR, limit)

    if filtering and k > top_k:
        log.debug(
            "search_vector.widened top_k=%d final_k=%d kept=%d", top_k, k, len(rows[:top_k])
        )

    return [
        SearchHit(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            path=row["path"],
            title=row["title"] or row["path"],
            heading_path=row["heading_path"] or "",
            prelude=row["prelude"] or "",
            body=row["body"] or "",
            sensitivity=row["sensitivity"],
            score=row["score"],
            source="vector",
            distance=row["score"],
        )
        # Widening can overshoot — vec0 returned more permitted rows than asked
        # for. Truncate so the contract ("at most top_k") holds either way.
        for row in rows[:top_k]
    ]


#: Standard RRF damping constant — large enough that rank 1 vs rank 2 isn't a
#: cliff, small enough that being in both lists still clearly outranks being
#: high in just one. Not config: it is a property of the fusion formula, not a
#: deployment choice, and every source list already has its own configurable
#: width (retrieval.vector_top_k / fts_top_k).
_RRF_K = 60


def fuse_rrf(
    vector_hits: list[SearchHit], fts_hits: list[SearchHit], *, top_k: int
) -> list[SearchHit]:
    """Reciprocal rank fusion: combine by *position* in each list, not by the
    raw score — bm25 and vector distance are not on comparable scales, so
    anything that tried to add them together would be combining noise.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, SearchHit] = {}
    #: Which arms found each chunk. Fusion is where origin used to be lost —
    #: the merged hit simply kept whichever list happened to be iterated first
    #: — so being in both lists (the thing RRF most rewards) was invisible
    #: afterwards. Tracked in insertion order so "vector+fts" reads the same
    #: way every run.
    sources: dict[str, list[str]] = {}
    for hits in (vector_hits, fts_hits):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            by_id.setdefault(hit.chunk_id, hit)
            found_in = sources.setdefault(hit.chunk_id, [])
            if hit.source and hit.source not in found_in:
                found_in.append(hit.source)

    ranked = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
    return [
        replace(by_id[chunk_id], score=score, source="+".join(sources[chunk_id]))
        for chunk_id, score in ranked
    ]


def _lead_chunk(
    conn: sqlite3.Connection, doc_id: str, *, sensitivity: str | None
) -> SearchHit | None:
    """A doc's first chunk (ordinal 0), standing in for the whole note in
    graph expansion — cheaper than deciding which of a note's chunks is
    "the" relevant one when the note was reached by a link, not a match.
    """
    row = conn.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.heading_path, c.prelude, c.body, c.sensitivity,
               d.path, d.title
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.doc_id = :doc_id AND (:sensitivity IS NULL OR c.sensitivity = :sensitivity)
        ORDER BY c.ordinal
        LIMIT 1
        """,
        {"doc_id": doc_id, "sensitivity": sensitivity},
    ).fetchone()
    if row is None:
        return None
    return SearchHit(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        path=row["path"],
        title=row["title"] or row["path"],
        heading_path=row["heading_path"] or "",
        prelude=row["prelude"] or "",
        body=row["body"] or "",
        sensitivity=row["sensitivity"],
        score=0.0,
    )


def expand_graph(
    conn: sqlite3.Connection,
    hits: list[SearchHit],
    *,
    max_siblings: int,
    discount: float,
    sensitivity: str | None = None,
) -> list[SearchHit]:
    """One-hop graph expansion (README "Retrieval"): for each hit's doc, pull
    the topic note(s) it belongs to and up to `max_siblings` other docs that
    same topic note also links to. Returns new candidates only — a doc already
    present among `hits` is never duplicated — each scored at a fixed
    `discount` off the hit that reached it, so an expanded chunk wins only
    when genuinely competitive with the direct hits, not merely present.
    Expects `hits` already fused (a positive, higher-is-better score) — see
    vault_ask.ask, which never calls this on raw single-source hits.

    One hop, strictly: expansion walks outward from `hits` only, never from a
    doc this call itself just added — that recursion is what would let the
    graph hop swamp the results (README "Retrieval").
    """
    if not hits:
        return []

    present_docs = {h.doc_id for h in hits}
    expanded: dict[str, SearchHit] = {}

    for hit in hits:
        topic_docs = [
            row["src"]
            for row in conn.execute(
                "SELECT DISTINCT src FROM edges WHERE kind = 'topic' AND dst = ? AND resolved = 1",
                (hit.doc_id,),
            ).fetchall()
        ]
        for topic_doc in topic_docs:
            topic_lead = _lead_chunk(conn, topic_doc, sensitivity=sensitivity)
            # A topic hub the caller may not see is not just withheld — it is
            # not traversed. Pulling its open members in anyway would leak the
            # hub by inference: the citation set would quietly encode "these
            # notes co-belong to a `personal` topic page".
            if topic_lead is None:
                continue
            if topic_doc not in present_docs and topic_doc not in expanded:
                expanded[topic_doc] = replace(
                    topic_lead, score=hit.score * discount, source="graph"
                )

            siblings = conn.execute(
                # Sensitivity is filtered here in SQL, before LIMIT — unlike
                # the vec0 path (see search_vector), an ordinary join CAN do
                # that. Filtering after LIMIT would mean a topic whose first
                # `max_siblings` members happen to be personal expands to
                # nothing while permitted siblings sit right behind them.
                # ORDER BY for determinism: LIMIT without it made *which* 5 of
                # a 35-member topic you got depend on scan order, so results
                # could shift after an unrelated reindex.
                "SELECT DISTINCT e.dst FROM edges e "
                "JOIN docs d ON d.doc_id = e.dst "
                "WHERE e.kind = 'topic' AND e.src = :topic AND e.resolved = 1 "
                "  AND e.dst != :exclude "
                "  AND (:sensitivity IS NULL OR d.sensitivity = :sensitivity) "
                "ORDER BY e.dst "
                "LIMIT :max_siblings",
                {
                    "topic": topic_doc,
                    "exclude": hit.doc_id,
                    "sensitivity": sensitivity,
                    "max_siblings": max_siblings,
                },
            ).fetchall()
            for row in siblings:
                sibling_doc = row["dst"]
                if sibling_doc in present_docs or sibling_doc in expanded:
                    continue
                lead = _lead_chunk(conn, sibling_doc, sensitivity=sensitivity)
                if lead is not None:
                    expanded[sibling_doc] = replace(
                        lead, score=hit.score * discount, source="graph"
                    )

    return list(expanded.values())


def assemble_context(hits: list[SearchHit], *, sensitivity: str | None = None) -> str:
    """Chunks rendered for the prompt, each labelled with its citation.

    Score order. Graph order (README "Answering contract" — topic note leads,
    chunks from one note stay adjacent) is still unimplemented; the edges table
    it needs now exists, so this is a real gap rather than a blocked one.

    ``sensitivity``, when given, is asserted rather than applied: this is the
    last point before text becomes a prompt, and the invariant that no
    `personal` chunk gets here otherwise rests on four separate retrieval call
    sites each having been passed the right argument. Cheap to check once at
    the boundary; a silent leak is the one bug in this project that cannot be
    walked back.
    """
    if sensitivity is not None:
        leaked = [h.path for h in hits if h.sensitivity != sensitivity]
        if leaked:
            raise AssertionError(
                f"sensitivity gate breached: {len(leaked)} chunk(s) not {sensitivity!r} "
                f"reached context assembly: {sorted(set(leaked))[:5]}"
            )
    parts = []
    for hit in hits:
        citation = f"[[{hit.path}|{hit.title}]]"
        heading = f" ({hit.heading_path})" if hit.heading_path else ""
        parts.append(f"### {citation}{heading}\n{hit.body}")
    return "\n\n".join(parts)
