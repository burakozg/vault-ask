# vault-ask

Answers questions against the Obsidian vault, using the vault as the authoritative
source and the web only as a labelled fallback. Semantic retrieval and the link
graph together — a question about Anthropic should reach the topic note, the nine
clippings under it, and the podcast episodes that mention it, because those are
already connected in the vault.

It is the **fifth** application on this vault and the **first that does not write
to it**. `taster`, `podcast-digest`, `security-digest` and `clippings-topics` all
hold a region in somebody's note. This one holds nothing. The vault is opened
read-only; there is no merge path, no marker pair, no frontmatter key. If a
future feature wants to persist an answer, it writes to its own store, not to
`99 topics/`.

```sh
./deploy                                   # build, ship, run it on the NAS
python -m vault_ask index                  # incremental index build
python -m vault_ask index --rebuild        # full re-embed
python -m vault_ask index --dry-run        # report what would change; write nothing
python -m vault_ask ask "what have I read about Entra ID?"
python -m vault_ask ask "..." --dry-run    # retrieval only, no generation
python -m vault_ask serve                  # FastAPI: OpenAI shim first; REST/MCP later
```

`ask` and `serve` land in later build steps below — see "Build order" for
what exists right now.

## Architecture

One core, three adapters. The adapters must contain no retrieval logic — if a
behaviour cannot be exercised by `python -m vault_ask ask`, it is in the wrong
layer.

```
                      ┌─────────────────────────────┐
  CouchDB (LiveSync) →│  ingest → chunk → embed     │→ SQLite (chunks, vec, fts, edges)
                      └─────────────────────────────┘
                                                          ↑
                      ┌─────────────────────────────┐     │
                      │  retrieve → expand → rerank │─────┘
                      │  → assemble → answer        │→ litellm → any model
                      └─────────────────────────────┘
                             ↑          ↑         ↑
                        REST JSON   OpenAI shim   MCP
```

| adapter | path | what it is for | status |
|---|---|---|---|
| OpenAI-compatible | `POST /v1/chat/completions` | the built-in chat UI at `/chat`, and any OpenAI-speaking client | ✅ built |
| MCP | `/mcp` (streamable HTTP) | Claude Code and Claude Desktop, as tools | ✅ built |
| REST | `POST /ask`, `POST /search`, `GET /graph/{slug}` | other homelab apps, scripts | future work |

**The OpenAI shim was built first**, ahead of REST and MCP, because it was the
only adapter of the three that came with a UI attached — point any OpenAI-speaking client at it
and there is a chat interface, with no frontend of this project's own to write.

That reasoning held right up until the UI was measured. Open WebUI cost a
5.09 GB image and **~15m35s of cold Python imports on every container
recreate** — torch, sentence-transformers, langchain, chromadb — for a
retrieval stack that is dead weight here, because vault-ask *is* the retrieval
layer. What it actually supplied was a chat box. So there is one at `/chat`
now, served by this app: a single self-contained HTML file in the same style as
`/admin` — no CDN, no build step, no second container, no second database. The
shim was still the right first adapter; the free UI just stopped being free.
See "Deployment" for the measurement. MCP followed once there was a
concrete user for it (Claude Code, in this very session). REST exists to let
other homelab apps call this one programmatically and is still deferred until
one of them actually asks for it — built speculatively, it would be a guess
at a shape nothing has validated yet. The core (retrieve → expand → assemble →
answer — rerank is still unbuilt, see "Later") is adapter-agnostic either way,
so REST is new adapter code against an unchanged core whenever it lands, not a
redesign.

The OpenAI shim advertises a single model id, `vault-ask`. Streaming is real
token streaming from the generation model, not a pre-computed answer typed out
— time-to-first-token is the point, and `/chat` renders the SSE deltas as they
arrive.

`/chat`'s one non-obvious control is the sensitivity switch, labelled **include
personal notes** rather than by the wire parameter. `allow_web` inverts:
`allow_web: true` *narrows* retrieval to `open` chunks. A control whose name is
the opposite of its effect is one people get backwards, and getting this one
backwards puts personal notes into a web-capable context — so the UI names the
decision it makes, and `tests/test_openai_shim.py` pins the inversion.

MCP exposes retrieval as tools rather than one `ask` tool, because the calling
model can then do its own multi-hop: `vault_search(query, k)`,
`vault_read(path)`, `vault_neighbors(path)`, `vault_topics()`. Only
`vault_search` is the RAG entry point; the other three are why the graph
exists — a model that gets a `vault_search` hit can follow it to the whole
note, its neighbours, or its topic page, the same way a person would in
Obsidian. See `vault_ask/api/mcp_adapter.py`.

## Choosing a chat UI

Two are deployed right now, on purpose, so the choice can be made by using them
rather than by reasoning about them:

| | `/chat` (built in) | NextChat | ~~Open WebUI~~ |
|---|---|---|---|
| where | `http://APP_LAN_IP:8080/chat` | `http://WEBUI_LAN_IP:3000` | removed |
| image | none — one 11 KB page | 196 MB (63 MB compressed) | 5.09 GB (1738 MB compressed) |
| restart → reachable | n/a, part of vault-ask | **31 s** | **16.2 min** |
| memory | — | 43 MiB | 842 MiB |
| state | browser localStorage | browser localStorage | SQLite DB, accounts, migrations |
| upstream | this repo | **stale: last image 2025-07-29** | actively maintained |

Open WebUI was the original choice and is gone. It cost 5.09 GB and ~15m35s of
cold Python imports per recreate — torch, sentence-transformers, langchain,
chromadb — for a retrieval stack that is dead weight here, because vault-ask
*is* the retrieval layer. Its `webui-data/` directory is deliberately left on
the NAS, so reinstating it is a compose edit and not a restore.

### What the survey found

Worth recording, because it is most of the argument:

- **NextChat** — 63 MB compressed, verified working against vault-ask end to
  end. But the newest image under the reachable name is `v2.16.1`, pushed
  **2025-07-29**; the project renamed and its new images are not pullable from
  here. Deployed anyway, because a working stale thing beats an unverified
  fresh one for a comparison — and because the staleness *is* a data point.
- **Hollama** — smaller still (54 MB compressed), but its tags are incoherent
  (`1.0.6` is from 2024, *older* than `:latest`), and nothing in the shipped
  bundle advertises OpenAI-compatible support. Not deployed: unverifiable.
- **LibreChat** — 832 MB and requires MongoDB and Meilisearch. Not thin.

The pattern: in this space you get **actively maintained but enormous**, or
**thin but unmaintained**. That is the strongest argument for `/chat`, whose
upstream is this repository and which cannot go stale independently of it.

### What each is actually for

`/chat` is 11 KB, starts with vault-ask, stores conversations in
localStorage, and knows about this application specifically — the sensitivity
switch is labelled *include personal notes* rather than `allow_web`, and
`[[wikilink|citations]]` render as titles with the path on hover. A generic
client cannot do either; it does not know what those mean.

NextChat is the control: a real third-party product, with prompt templates,
multiple conversations, better mobile handling and a polished model picker.
If those matter more than the vault-specific affordances, that is the answer.

Nothing is lost by keeping both — the UI is not where the value is. Both talk
to the same `/v1/chat/completions`, so anything either can do, any
OpenAI-speaking client can do.

## Configuration

Layered like `podcast-digest`'s: a non-secret `config.yaml` (committed, the
deployment default) plus environment variables for secrets and per-deployment
overrides — `VAULTASK_<SECTION>__<KEY>` (double underscore nests), e.g.
`VAULTASK_RETRIEVAL__FINAL_TOP_K=10`. Every section is a strict pydantic model
(`extra="forbid"`): a typo'd key in `config.yaml` is a startup crash, not a
silent no-op. See `vault_ask/config.py` for the full schema and
`.env.example` for the secrets it reads.

This departs from the shape sketched in early design notes for this project
(TOML, `[corpus]`/`[sensitivity]`/`[models]` tables) in favour of the format
every other app in this stack already uses — one less thing to hold in your
head across five repos.

## Admin console

`GET /admin` — a browser-based console for the settings worth changing without
a `config.yaml` edit and a redeploy: **which model answers** (a type-ahead over
a curated shortlist, but still free text — any litellm-routable id works),
**which search provider** runs the web fallback (`duckduckgo` / `tavily` /
`brave`), and `retrieval.*` tuning (`graph_enabled`, `vector_top_k`,
`fts_top_k`, `fusion_top_k`, `graph_max_siblings`, `graph_discount`,
`graph_max_slots`, `final_top_k`).

**Provider keys are not settable here, deliberately.** Secrets in this project
are environment-only — never in `config.yaml`, never in `overrides.json`, never
in a browser form. DuckDuckGo needs no key and is the default, so the feature
works with nothing to sign up for. Tavily and Brave appear in the dropdown but
are **disabled until their key exists**, and selecting one anyway is rejected
with the env var named:

> `'tavily' has no API key configured, so selecting it would silently return no
> results. Set VAULTASK_TAVILY_API_KEY in .env and redeploy, then choose it here.`

Refusing beats saving: a stored-but-unusable provider looks fine in the console
and then returns nothing after the next restart, with no symptom pointing at
the cause.

The model shortlist (`config.GENERATION_SUGGESTIONS`) was checked against
OpenRouter's live catalogue rather than recalled — the failure mode a
hand-written list invites — and it will still go stale; re-check with
`curl -s https://openrouter.ai/api/v1/models | jq -r '.data[].id'`. A live
fetch was considered and rejected: it puts a network call and an outage mode
into a page whose job is to work when things are broken. Same shape as `podcast-digest`'s own admin console: a single
self-contained HTML file (no CDN, no build step), the admin key entered once
and held in `sessionStorage`, sent as `X-API-Key` on every request to
`/admin/config`. The page itself is served unauthenticated — only the data
behind it is gated (`vault_ask/api/admin_auth.py`, `hmac.compare_digest`,
10 failures per address before a 429, unset key fails **closed** with 503,
never open).

Deliberately excludes `models.rerank` and `retrieval.rerank_top_k`: no rerank
step is wired into the pipeline yet (see "Build order" below), and a console
knob that changes nothing would be a UI that lies. Deployment topology
(`vault.*`, `models.embedding*`, `api.*`, `index.*`) and every secret stay
file/environment-only for the same reason podcast-digest's own console
excludes them — a typo in a browser form must not be able to point the vault
connection somewhere wrong or lock the container out of its own port.

**Applies on restart, not live.** `Settings` is built once, at process start,
and `app.state.cfg` is a reference every request handler already holds — a
"live" edit would only be true for readers that happen to re-check. Saved
changes are stored as a small JSON file at `VAULTASK_OVERRIDES_FILE` (default
`/data/overrides.json`, the same bind-mounted, writable directory the SQLite
index lives in) and layered into the settings chain **below the environment
but above `config.yaml`** (`vault_ask/config.py::_OverridesSource`) — an
override beats the shipped default without an image edit, but deployment
topology set via `VAULTASK_*` env vars still wins over a browser edit. A
write is validated by merging it onto the process's *currently active*
values for the fields that stay file-only and re-running the same pydantic
model config.py itself uses (`ModelsConfig` / `RetrievalConfig`), so e.g. an
override that would violate `retrieval.rerank_top_k >= final_top_k` against
the real, running `rerank_top_k` is rejected at save time, not discovered as
a startup crash after the restart. `GET /admin/config` reports
`pending_restart`: whether `overrides.json`'s current content differs from
what this process actually booted with, not whether it differs from the
shipped defaults — so restarting into an override that exactly restates
`config.yaml` correctly reports nothing pending.

Set `VAULTASK_ADMIN_API_KEY` (`.env.example`) to a long random value
(`openssl rand -hex 32`) to enable it — `vault_ask/api/admin.py`,
`vault_ask/api/admin_auth.py`, `vault_ask/overrides.py`,
`vault_ask/api/static/admin.html`.

## Corpus and sensitivity

The whole vault is indexed. Not all of it may leave the house.

Every chunk carries a `sensitivity` of `open` or `personal`. The classification is
config, not inference:

```yaml
corpus:
  include: ["**/*.md"]
  exclude: ["00 inbox/**", "**/.trash/**"]

sensitivity:
  personal_paths: ["Tastings/**", "30 projects/**"]
  frontmatter_key: sensitivity   # per-note override, wins over path
```

The rule that this buys, and the one the tests exist to defend:

> **A query that will touch the web is answered from `open` chunks only.**
> `personal` chunks are never placed in a context window that also carries a web
> tool, and their text is never used to formulate a search query.

So the pipeline decides web-augmentation *before* retrieval, not after. The
request carries `allow_web: bool` (default true), and `allow_web` narrows the
retrieval filter to `sensitivity = 'open'`. A local-only question — `allow_web:
false` — sees the whole vault. Getting this ordering backwards is the one bug in
this project that leaks something.

MCP and the OpenAI shim both default to `allow_web: false`, because in both cases
a model you do not control is holding the transcript.

### Two ways this failed silently, and what now catches them

Both were found by auditing the *running* deployment rather than the code, and
both are the same shape: a privacy control that is invisible when broken.

**1. The patterns matched nothing.** This shipped with `personal_paths` of
`30 journal/**`, `40 people/**`, `50 tastings/**` against a vault whose real
folders are `30 projects/` and `Tastings/`. There is no `40 people/` at all,
and matching is `fnmatchcase` — **case-sensitive**, against the note's real
`path`, not its lowercased CouchDB `_id`. Result: all 2,254 notes classified
`open`, and the entire `allow_web` gate guarded an empty set. Nothing failed;
there was simply never anything to filter. `ingest.run_ingest` now warns on
every `personal_paths` pattern matching zero candidates.

Sensitivity is also **recomputed on every index run**, not only when a note
changes. Change detection is rev-based, so after a config edit every doc is
`unchanged`, nothing is re-read, and the new rule would silently not apply —
the control would look fixed and not be. The recompute is a plain `UPDATE` on
`docs` and `chunks`, deliberately *not* routed through `replace_chunks`, which
would drop the doc's rows from `chunks_vec` and re-embed the corpus to change
one column. Measured: 87 docs reclassified with all 5,664 embeddings intact.

**2. Vector search filters *after* `k`, not before.** `chunks_vec` is
`vec0(chunk_id, embedding)` — it carries no `sensitivity` column, so vec0
cannot see the predicate. `k = :top_k` binds first and `c.sensitivity` is
evaluated on the rows vec0 already chose. Reproduced on this project's own
sqlite-vec 0.1.9: with 10 personal chunks nearer the query than 3 open ones,
`top_k=5, sensitivity='open'` returned **zero** rows while open matches
existed. FTS5 and the graph-sibling query filter in the WHERE clause before
`LIMIT` and were always correct; only the KNN path is affected.

Nothing leaked — the filter does apply — but the vector arm went silently
short, and RRF then fused a full-width FTS list against a stunted vector list,
tilting answers toward keyword matching exactly when the question was near
personal material. `search_vector` now widens `k` and re-asks until it has
`top_k` permitted rows or vec0 runs out of corpus.

`tests/test_sensitivity.py` is where this is defended. The distinction it
exists for: asserting "personal text is absent from the output" passes under a
broken filter-after implementation too. The tests that matter seed *more
personal candidates than the `k` being asked for*, so `k` genuinely binds, and
assert the permitted result count is **preserved** — that a personal chunk
never consumes a slot. Each was verified to fail against the pre-fix code.

Metadata is gated at the same choke point as text. `_resolve_doc` in the MCP
adapter filters on `sensitivity` during *resolution*, so a note the caller may
not see does not resolve at all: it cannot be confirmed to exist, cannot have
its real path echoed back by an "ambiguous — matches: ..." message, and cannot
have its title or link structure returned by `vault_neighbors`. Filtering only
at the point of returning a body left all of those open, which it previously
did — including a refusal message that named the very note it was refusing.

## Web fallback

The vault stays the authoritative source; the web supplements it when the vault
is thin. **Off by default in code** — an operator who has not thought about it
gets a vault-only system — and on in this deployment's `config.yaml`. Toggle it
at `/admin` without a redeploy.

```yaml
web:
  enabled: true
  max_results: 3        # small on purpose: a long list of snippets is how the
                        # web stops being a supplement and starts being the answer
  thin_hits: 3
  thin_distance: 1.0    # measured, see below
```

### The invariant

Web search can only ever run when `allow_web` is true — the same flag that
restricts retrieval to `open` chunks. That is the entire design: **web content
and `personal` content can never be in one context, because the single switch
that admits one excludes the other.** It is structural, not a matter of care at
each call site, and `tests/test_web.py::TestNeverWithPersonalContext` fails if
anyone decouples them.

What it does *not* protect: **your typed question goes to the search provider
verbatim.** No gate covers that and none can — it is what searching means. The
README's rule that personal *chunk text* never formulates a query holds
automatically, since personal chunks are not retrievable in this mode; your own
words are a different matter, and worth knowing rather than implying the switch
makes search private.

### When it fires: measured, not guessed

"Thin coverage", not "vault silent". With ~2,250 notes FTS returns *something*
for nearly any question, so a silence trigger would essentially never fire.

The first attempt thresholded the fused RRF score and **could not work**: RRF is
rank-based, so the top hit scores `1/61 = 0.0164` whether the match is perfect
or nonsense. Measured, *"what is the airspeed velocity of an unladen swallow?"*
scored exactly 0.0164 against this vault — identical to a well-covered
question. The fused score carries no relevance information at all.

Vector distance does, and it separates cleanly. `SearchHit.distance` now
survives fusion so the trigger can read it:

| | best vector distance |
|---|---|
| covered by the vault | 0.815 – 0.955 |
| not covered | 1.022 – 1.129 |

`thin_distance: 1.0` sits in that gap. Over 12 questions (6 covered, 6 not),
**0 misclassified**. Re-measure after a corpus change or an embedding swap:

```sh
uv run python scripts/measure_web_trigger.py
```

It reports the two distributions, whether they separate, and where the
threshold should sit — and says so loudly if the sets overlap, in which case
the signal needs rethinking rather than the number nudging.

Known limitation: the trigger is per-question, not per-sub-question. Ask
something the vault mostly covers and it will not reach out for the one missing
fact — *"what is MCP and who created it?"* answers the first half from the vault
and says the vault does not cover the second, rather than searching.

### What an answer looks like

The separation contract in `vault_ask/prompts.py` was written for this and sat
unused until now. Real output, `web.enabled: true`:

> The vault does not contain information on how to make a sourdough starter
> from scratch. However, I can provide information from the web on this topic:
>
> **From the web:**
> * **The Clever Carrot** suggests … [https://www.theclevercarrot.com/…]

Vault silence stated first, web under its own heading, cited by URL — never as
a wikilink, because a wikilink asserts the user has a note they do not have.

`ddgs` (DuckDuckGo) is deliberately the only unofficial interface in this
project: no account, no key, and it scrapes, so it is rate-limited and will
eventually break. That is an accepted trade because web results are
best-effort by construction — a breakage degrades an answer to vault-only
rather than failing the request (`tests/test_web.py::TestDegradesGracefully`).

## Ingestion

Same source of truth and the same change detection as `clippings-topics`, which
is where the non-obvious parts were already paid for:

```
GET /tastings/_all_docs?startkey="..."&endkey="...0"&include_docs=true
```

`include_docs` is a correctness requirement. **LiveSync does not tombstone a
deleted note** — the CouchDB document stays live with `deleted: true` in the
body, so a row listing includes notes that exist on no device. Indexing those
produces an assistant that cites files the user cannot open, which is worse than
missing them. `vault_ask/vault.py` filters them out of every listing.

Change detection is a diff against the `docs` table itself — `doc_id → (rev,
content_hash)` — rather than a separate cache file, since the whole point of
having a SQLite index is that it already is the cache:

| in listing | in cache | |
|---|---|---|
| rev differs, hash differs | yes | re-chunk, re-embed, replace all chunks for that doc |
| rev differs, hash same | yes | touch cache only — no embedding spend |
| — | no | new — chunk and embed |
| gone / deleted | yes | delete its chunks and its outbound edges |
| rev same | yes | untouched |

Chunk replacement is **delete-then-insert for the whole document**, never a
per-chunk diff. Same reasoning as rebuilding the marker region whole in the
writer apps: an edit can remove a section, and only a full replacement makes its
chunk disappear.

`--rebuild` disables the `touched`/`unchanged` shortcuts — every candidate is
read and reclassified — but still diffs the real cache for deletions, so a
rebuild that finds the vault has shrunk still notices.

## Chunking

Heading-aware, not fixed-window. Split on `##`/`###` boundaries, target ~1000
tokens, merge runts into the previous chunk, hard-split anything over 2000.

Every chunk is stored with a **prelude** that is embedded along with the body:
the note title, its path, and its topic memberships. A chunk from the middle of a
clipping about model routing is nearly unretrievable on its own; the same chunk
prefixed with `Anthropic — The Complete Claude Code Setup` is not. Store the
prelude separately from the body so it can be stripped before display.

Chunk ids are `sha256(doc_id + heading_path + ordinal)[:16]`, stable across
re-runs of unchanged input.

## Index

One SQLite file. No graph database — the graph here is a few thousand edges, and
a join beats an operational dependency.

```sql
CREATE TABLE docs (
  doc_id TEXT PRIMARY KEY,       -- vault path, lowercased
  title TEXT, rev TEXT, content_hash TEXT,
  sensitivity TEXT NOT NULL,     -- 'open' | 'personal'
  mtime TEXT, frontmatter JSON
);

CREATE TABLE chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
  ordinal INTEGER, heading_path TEXT,
  prelude TEXT, body TEXT,
  sensitivity TEXT NOT NULL      -- denormalised from docs; see "Sensitivity" on
                                 -- why this is filter-before for FTS but not vec0
);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
  chunk_id TEXT PRIMARY KEY, embedding FLOAT[1024]
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  chunk_id UNINDEXED, prelude, body, tokenize='unicode61'
);

CREATE TABLE edges (
  src TEXT NOT NULL,             -- doc_id
  dst TEXT NOT NULL,             -- doc_id, or unresolved target
  kind TEXT NOT NULL,            -- 'wikilink' | 'topic' | 'tag'
  resolved INTEGER NOT NULL,
  PRIMARY KEY (src, dst, kind)
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- embedding_model, embedding_dim, schema_version, last_run
```

`meta.embedding_model` is checked on every run. A changed embedding model or
dimension forces `--rebuild` and says so; silently mixing vector spaces produces
retrieval that is wrong in a way no test catches.

Edges come from three places, all of which already exist because the writers put
them there:

- `[[path/to/note|alias]]` in any note body. `clippings-topics` writes every link
  path-qualified, which is what makes resolution deterministic rather than a
  guess between three notes sharing a basename.
- membership in a `99 topics/` note's `<!-- begin:clippings -->` region → `topic`
  edges from the topic to each clipping.
- frontmatter `tags` → `tag` edges, for the reader's own taxonomy.

Unresolved links are stored with `resolved = 0` rather than dropped; they are a
useful signal later ("things I keep referring to but never wrote up").

## Retrieval

```
question
  → (if allow_web) restrict to sensitivity='open'
  → vector top-40  ─┐
  → FTS5 top-40   ──┼→ reciprocal rank fusion → top-20
  → graph expansion: for each hit's doc, pull its topic notes and up to N
    sibling docs sharing a topic edge; score them at a discount
  → rerank top-30 → top-8, of which at most graph_max_slots are expanded
  → assemble context, generation
```

The graph hop is the part that a plain vector store cannot do and the reason this
project is worth building over the vault rather than over a folder of markdown.
Three rules keep it from swamping the results: expansion is one hop only,
expanded chunks enter at a fixed discount, and **at most `graph_max_slots` of
the final answer may be expanded chunks**.

### Graph expansion, measured

The third rule exists because the first two were not enough, and the README
previously claimed they were. Run it yourself:

```sh
uv run python scripts/measure_graph_expansion.py            # A/B table
uv run python scripts/measure_graph_expansion.py --verbose  # per-hit detail
```

It needs a populated index and a reachable Ollama host, so it is a script
rather than a test — same reasoning as `podcast-digest`'s
`scripts/check-enclosure-chains.py`. The A/B is possible at all only because
`SearchHit.source` now records provenance and `retrieval.graph_enabled` is a
real off switch; `graph_max_siblings: 0` is *not* one, since the topic-note
pull runs independently of the sibling query.

**The discount does not do what the README said it did.** With `_RRF_K = 60`, a
hit ranked 1 in both arms scores `2/61 = 0.0328`, so a chunk it pulls in scores
`0.0328 × 0.7 = 0.0230` — above a rank-1 *single-arm* hit at `1/61 = 0.0164`.
An expanded chunk, never scored against the question at all, could outrank a
genuine top hit. Measured over 20 questions (half on subjects with topic notes,
half without):

| | before | after |
|---|---|---|
| questions where expansion fired | 9/20 (45%) | 9/20 (45%) |
| expanded chunks in final answers | 36 | 18 |
| direct hits they displaced | 36 | 18 |
| worst case, one question | 6 of 8 slots | 2 of 8 slots |

Displacement was **1:1** — every expanded chunk evicted a directly-scored one.
Expansion was substitutive, not additive.

**It was still worth keeping.** The per-hit detail shows why the call is not
obvious: on *"what have I read about AI agents?"* expansion surfaced
`99 topics/ai-agents.md` plus four on-topic agentic notes, displacing a
tangential security survey and three near-duplicate chunks of one unrelated
note — a clear win. On *"Claude and Anthropic"* it evicted three daily-digest
items literally about Anthropic in favour of a GPT-5.5 evaluation and
"What Boards Must Demand" — a clear loss.

So the fix is a quota, not score tuning. There is nothing to tune: every
expanded chunk inherits the *same* score from the hit that reached it, so they
tie exactly and their order among themselves is arbitrary. `graph_max_slots`
(default 2 of 8) is a **ceiling, not an allocation** — expanded chunks still
have to out-score a direct hit to appear at all, which is why selectivity stayed
at 45% rather than becoming 100%. The one exception: if there are not enough
direct hits to fill the answer, expansion may fill past the quota, since that
thin-answer case is what it was introduced for.

**An unrelated problem this surfaced.** Among the chunks expansion displaced
were three near-duplicate chunks of the *same* note occupying 3 of 8 slots.
Base retrieval has no per-doc diversity cap, and expansion was partly papering
over that. Worth fixing on its own terms — see "Open questions".

Reranking uses `google/gemini-2.5-flash-lite` via OpenRouter by default — already
measured on this corpus (by `clippings-topics`) at ~1s with real answers, where
`qwen3.7-flash` 429s under any load. A local `bge-reranker-v2-m3` is the
pluggable alternative.

## Answering contract

The system prompt is code, not a config string, and it is tested.

- Every factual claim carries a citation as a wikilink to the source note:
  `[[10 raw/Claude/The Complete Claude Code Setup for 2026 Every Skill|…]]`.
  Path-qualified, so it resolves when pasted into Obsidian.
- Vault content and web content are **visually separated** in the answer. Web
  claims are prefixed and their source is a URL, never a wikilink.
- Where the two disagree, the answer says so and does not adjudicate.
- If the vault has nothing, the answer says the vault has nothing, and only then
  offers the web. "The vault is silent on this" is a correct answer and must not
  be trained out by a helpful-sounding prompt.

Context assembly places chunks in graph order, not score order — chunks from the
same note stay adjacent and the topic note leads. Models summarise a coherent
document better than a shuffled pile of passages.

## Models

Everything through `litellm`. Nothing else in the codebase knows a provider name.

```yaml
models:
  generation: openrouter/google/gemini-2.5-flash
  rerank: openrouter/google/gemini-2.5-flash-lite
  embedding: ollama/bge-m3
  embedding_base_url: null       # deployment-set — see OLLAMA-SETUP.md
  embedding_dim: 1024
```

**Embedding runs local, decided.** It is the one call that touches every note
in the vault, including the `personal` ones, on every `--rebuild`, so it never
goes to a cloud endpoint regardless of what `generation`/`rerank` are pointed
at. `embedding_base_url` is deployment topology, the same way podcast-digest's
`asr.remote_url` is: no committed default, set per-deployment via
`VAULTASK_MODELS__EMBEDDING_BASE_URL` to whichever LAN machine runs Ollama. See
`OLLAMA-SETUP.md` for getting that machine ready — pull `bge-m3`, make Ollama
listen on the LAN rather than only localhost, and verify the address is
reachable before pointing vault-ask at it. Multilingual matters here — the
vault carries Turkish and Swedish, which is part of why `bge-m3` specifically.

The generation model is a config line so the hardware decision does not block
this. When local 27–32B inference lands, `generation` points at Ollama and
nothing else changes; the OpenRouter path stays as the fallback for the rerank
step, where latency per call matters more than privacy.

## Deployment

✅ Compose project on the NAS, deployed over ssh — see `homelab/README.md` for
the contract, `./deploy` for the verbs. This one **does** listen, on
`APP_LAN_IP`, and it needs a qnet address regardless: the vault is CouchDB on
another macvlan address and the NAS host cannot route to its own macvlan
children. MAC pinned per network, like every other project here.

Built with `uv` (locked, `uv.lock` committed) inside a `python:3.12-slim`
image — verified that `sqlite-vec`'s extension loading actually works in that
base image before shipping it, since that's exactly the kind of thing that
differs silently between a Mac dev machine and a Debian container. Deployed
for real, not just rendered: image built for `linux/amd64` (the NAS is x86,
this Mac is arm64 — `nas_ship_image` cross-builds), shipped over ssh, brought
up, and confirmed reachable and answering from another machine on the LAN.

**Open WebUI runs on the NAS too**, as a second service in the same
`docker-compose.nas.yml` — not on a dev machine, so it survives the same
reboots vault-ask does. It gets its own qnet address (`WEBUI_LAN_IP`) for a
browser to reach; the two containers reach *each other* over a private
`internal` bridge network by service name (`http://vault-ask:8080/v1`), never
through either qnet address — this NAS's macvlan has no embedded DNS and
container-to-container traffic between two macvlan addresses has been
observed to fail outright even by IP (`homelab/README.md`). Not marked
`read_only` like vault-ask's own container: it's a third-party image whose
full set of write paths hasn't been audited.

Recreating a container on this NAS's macvlan — even with its MAC correctly
pinned and unchanged — left both containers unreachable from the LAN for a few
minutes after this deploy, healthchecks notwithstanding ("Up (healthy)" is
about a container's own localhost, not the LAN). The fix is the one
`homelab`'s notes already describe: one outbound packet from inside the
container re-triggers ARP resolution on the router and it self-heals
instantly. What cost real time here was aiming that fix at a *guessed* gateway:
this qnet network is a `/25`, so its real gateway is not the `.1` a home
network's usual convention would suggest. Read it from
`docker network inspect <net> --format '{{(index .IPAM.Config 0).Gateway}}'`
rather than assuming — `vault_ask/api/app.py::_default_gateway` does the
in-container equivalent via `/proc/net/route`.

**Open WebUI's own SQLite hung on a warm restart** the second time this
container was recreated — 0.6% CPU, no forward log progress past "Will assume
non-transactional DDL.", 8.7GB of cumulative block I/O against a 684KB
database. Not the ARP issue above (this one never touches the network); the
`webui.db-shm`/`webui.db-wal` files from the previous container's shutdown
were the problem — deleting them (never `webui.db` itself, which holds the
actual data) and restarting the *same* container (`docker start`, not a
recreate) fixed it immediately, migrations and all, in seconds.

**That WAL explanation was wrong.** A later recurrence was diagnosed properly
and it is worth correcting here, because the wrong version costs an hour every
time it recurs. The same symptom came back — stalled at "Will assume
non-transactional DDL.", ~6GB of reads against a 692KB database — and this time:

- deleting `-shm`/`-wal` did **not** clear it;
- the database was verified healthy (`PRAGMA integrity_check` ok, 43 tables,
  accounts and chats intact) — not corruption;
- converting it to `journal_mode=DELETE` did **not** clear it, and Open WebUI
  put it straight back into WAL on the next boot, so that conversion is not
  even persistent;
- a throwaway container with a completely **empty** data directory hung at the
  same place, which rules the stored database out entirely.

The actual cause, from timestamps rather than inference. One start:

```
15:39:01  container starts
15:54:36  Open WebUI prints its banner          <- 15m 35s of nothing
15:54:49  GET huggingface.co  (model revision check)
15:54:50  loading cached SentenceTransformer
15:54:53  weights loaded
15:55:05  first HTTP 200
```

**Only 29 seconds of that is the application starting.** The other 15m35s
elapses before it prints a single line — it is Python importing Open WebUI's
dependency tree (torch, sentence-transformers, langchain, chromadb) cold, off a
contended NAS disk with an evicted page cache. Measured *warm*, inside the
already-running container: `import torch` 25.6s, `import sentence_transformers`
90.7s. Cold and under load, that becomes minutes. The ~7 GB of block reads is
the 5.09 GB image plus libraries being faulted in, not database I/O.

Two consequences worth knowing. It reaches **huggingface.co on every start** to
check the model revision (fast when cached, a startup dependency nonetheless).
And the model cache is 888 MB for a ~90 MB model — HuggingFace keeps every
format variant (safetensors, pytorch, onnx, openvino) though only one is
loaded; disk waste rather than startup cost.

So: before touching the database, check `free -m` and `/proc/loadavg`, and give
it ten minutes. Do not run a second Open WebUI container to test a theory while
the first is starting; that makes the memory pressure worse, and doing exactly
that is what turned a slow start into an apparent hang during this
investigation.

**The fix that actually mattered: Open WebUI no longer has `depends_on`.**
Everything else here makes the slowness predictable; this one stops paying it.
Compose cascades recreation to dependent services, so naming vault-ask in
`depends_on` meant every deploy of *vault-ask* recreated *Open WebUI* — and a
recreate is the 16-minute cold start above. Nothing is lost by removing it:
`depends_on` only orders startup, never waits for readiness, and Open WebUI
resolves `OPENAI_API_BASE_URL` lazily when a chat happens. Open WebUI now
restarts only when its own image or config changes, which — with the digest
pinned — means when you deliberately change it.

Four further changes, in `./deploy` and `docker-compose.nas.yml`:

- **Open WebUI is pinned by digest**, not the rolling `main` tag. Every deploy
  was silently also a version upgrade, so when it misbehaved there was no way
  to tell "my change broke it" from "the image moved" — which is most of why
  this took an hour. Same stance as the pinned `uv` release and the committed
  `uv.lock`: a deploy should change what you changed and nothing else. Upgrade
  deliberately with `docker buildx imagetools inspect ghcr.io/open-webui/open-webui:main`.
- **Its reachability wait is 20 minutes, not 60 seconds.** Measured twice:
  ~9 min after a `docker start`, and **16.2 min** after a compose recreate under
  load — and a deploy always recreates, so 16 is the number to size against. The
  60s default reported a hard failure twice for containers that were starting
  normally, and an intermediate 10-minute setting was *still* too short. That is
  worse than no check: it points the investigation at the wrong thing. vault-ask
  itself gets 3 minutes for the same reason — it missed 60s at load average 19.7.
  `nas_wait_reachable` already took a `tries` argument; this deploy simply never
  passed one.
- **The deploy wakes Open WebUI's ARP entry** after start. vault-ask does its
  own (`api/app.py::_wake_arp`); Open WebUI is a third-party image with nowhere
  to add that, so it is done from outside.
- **A pre-flight memory warning** when the NAS has under 1.5 GB free, so the
  slow start is expected rather than discovered.

`vault_ask/db.py` turns WAL mode on deliberately (two connections, one file)
and lives on the same class of mount, but nothing has reproduced any of this
against `index.sqlite`.

Index runs on start, then hourly. Cheap by design — an hour of no vault edits
costs one `_all_docs` request and no LLM calls. The very first run on a fresh
index is not cheap, though — with no cache yet, every note in the vault is
"new" and gets read individually (no concurrency), which took several minutes
against a ~2,250-note vault. `--rebuild` pays this same cost again by design
(README "Index").

State is `/data/index.sqlite`, bind-mounted. Losing it costs a full re-embed in
money, never in correctness.

**The embedding host is still whatever machine you set `EMBEDDING_HOST` to in
`.deploy.env`** (OLLAMA-SETUP.md) — the NAS deployment does not change that
decision, it just means the NAS container now reaches out to it over the LAN
instead of a process on the same machine reaching `127.0.0.1`. Worth knowing
if that machine is a laptop: it has to be awake and Ollama has to be bound to
the LAN interface (`OLLAMA_HOST=0.0.0.0:11434`), not the default
localhost-only bind — a real deploy surfaces that distinction immediately,
where same-machine local dev never does.

## Tests

Three that matter. The rest is cosmetic.

1. **`test_sensitivity.py`** (lands with the retrieval pipeline, step 3+) — a
   `personal` chunk never appears in a context assembled with `allow_web=True`,
   asserted at the assembly boundary rather than the prompt, and asserted again
   for the query-formulation path. This is the test that stops the failure that
   actually costs something. `test_ingest.py::TestSensitivity` covers the
   classification half of this today — what lands in `docs.sensitivity` — which
   is necessary but not sufficient.
2. **`test_deleted.py`** (folded into `test_ingest.py::TestDeleted` for now) — a
   doc with `deleted: true` in its CouchDB body is absent from `docs`, from
   `chunks`, and from every `edges.dst`. `vault_ask/vault.py::list_prefix`
   filters it at the source; `test_ingest.py` covers what happens once a doc
   already indexed disappears from a later listing.
3. **`test_citations.py`** (lands with the answering contract) — every wikilink
   an answer emits resolves to a `doc_id` present in `docs`. A hallucinated
   citation looks exactly like a real one until it is clicked.

```sh
uv run pytest tests/ -q
```

## Build order

Reordered from the original design to optimise for **something running soon**
over full capability — get a real chat UI answering real questions first, then
come back for the parts that make answers better rather than merely present.
Each step still ends somewhere useful; the "fast path" ones are what stands
between now and a working Open WebUI chat, the "later" ones are real work
deliberately deferred rather than dropped.

**Fast path — to a working chat UI:**

1. **`ingest` + `docs`/`chunks` tables + `--dry-run`.** ✅ No embeddings, no LLM.
   Correctness of change detection is verifiable here and nowhere later —
   `vault_ask/ingest.py`, `tests/test_ingest.py`.
2. **Chunking + FTS5.** ✅ `vault_ask ask` does keyword retrieval and generation,
   with the answering contract (citations, silence-is-correct) enforced as code
   from day one — `vault_ask/chunk.py`, `vault_ask/retrieval.py`,
   `vault_ask/ask.py`, `vault_ask/prompts.py`.
3. **Embeddings + `sqlite-vec` + RRF fusion with FTS5.** ✅ Local Ollama, always
   — see OLLAMA-SETUP.md. `ask` degrades to FTS-only automatically if the
   embedding host is unreachable or unconfigured, so this is additive, not a
   dependency the rest of the pipeline can be broken by. `meta.embedding_model`/
   `embedding_dim` are checked on every `index` run — a changed model without
   `--rebuild` is a loud failure (`EmbeddingSpaceChanged`), not silently mixed
   vector spaces. `vault_ask/embed.py`, `vault_ask/retrieval.py::search_vector,fuse_rrf`.
4. **FastAPI hosting the OpenAI-compatible shim** (`/v1/chat/completions`,
   `/v1/models`, streaming) **+ background indexing.** ✅ `python -m vault_ask
   serve` runs both: real hourly indexing on its own connection
   (`index.refresh_interval_s`) and the query API on another, WAL-moded so
   neither blocks or tears the other's view (`vault_ask/db.py`). Verified live
   end to end — real HTTP, real SSE framing, streamed generation actually
   token-by-token from litellm (not a pre-computed answer typed out
   artificially) — up to the point of needing a real OpenRouter key, which is
   a credentials matter, not a code one. Point Open WebUI at it: this is the
   "something running" milestone — a chat UI, backed by the real vault, with
   the real answering contract (citations, vault/web separation, sensitivity
   filtering) already enforced. `vault_ask/api/`.
5. **`edges` + one-hop graph expansion.** ✅ Three sources, all already in the
   vault: `[[wikilinks]]` anywhere in a body, the links inside a
   `<!-- begin:clippings -->` region (`topic` edges, from `clippings-topics`'
   own marker), and frontmatter `tags`. Resolution (matching a link's target
   to a real `doc_id`) happens twice: once inline during chunking against
   whatever's in `docs` so far, then again as a whole-table catch-up pass
   after the run commits — needed because a note early in a big batch can
   link to one indexed later in the same batch, so its target does not exist
   yet at extraction time. `expand_graph` pulls in a hit's topic note(s) and
   up to `graph_max_siblings` sibling docs per topic, each discounted by
   `graph_discount` off the hit that reached it, and only ever one hop —
   never recursing into an expanded doc's own edges. `vault_ask/edges.py`,
   `vault_ask/retrieval.py::expand_graph`. **Measured — see "Graph expansion,
   measured" below.** It is kept, with a slot quota it did not previously have.
6. **MCP adapter** (`/mcp`, streamable HTTP) **, mounted on the same app.** ✅
   `vault_search(query, k, allow_web)`, `vault_read(path, allow_web)`,
   `vault_neighbors(path, allow_web)`, `vault_topics(allow_web)` — retrieval
   exposed as tools rather than one `ask` tool, so the calling model (Claude
   Code, Claude Desktop) can do its own multi-hop: search, then follow a hit
   to the whole note, its graph neighbours, or the topic page it belongs to.
   `allow_web` defaults false on every tool, same reasoning as the OpenAI
   shim, and the gate applies uniformly — a `personal` note refuses
   `vault_read` exactly as it's excluded from `vault_search`, not just when
   reached by search. Mounting an MCP server inside an existing FastAPI app
   needs its ASGI lifespan combined by hand (`AsyncExitStack`) — `Mount()`
   alone does not run a sub-app's own lifespan, which is what starts its
   session manager. `vault_ask/api/mcp_adapter.py`.
7. **Admin console** (`GET /admin`, `/admin/config`). ✅ Browser-based config
   editor for `models.generation` and `retrieval.*` tuning, same shape as
   `podcast-digest`'s own — see "Admin console" above for the full design.
   `vault_ask/api/admin.py`, `vault_ask/overrides.py`.

**Later — real work, deliberately deferred:**

- The REST adapter (`/ask`, `/search`, `/graph/{slug}`) — build when another
  homelab app actually wants to call this one.
- ~~Web fallback~~ — **built** (see "Web fallback" above). It landed last, as
  planned, once the pipeline's behaviour was pinned by `tests/test_sensitivity.py`.

## Open questions

- Conversation memory: multi-turn needs query rewriting against history. Punt to
  after the REST/OpenAI-shim adapters exist; the shim gets the full transcript
  from the client either way.
- **No per-doc diversity cap in retrieval.** Found while measuring graph
  expansion: on one question, three near-duplicate chunks of the same note held
  3 of 8 slots — and the same note existed at two paths (`10 raw/Obsidian/…`
  and `10 raw/AI Tools/…`), so it was really one piece of writing taking 3/8 of
  the answer. Expansion was partly compensating for this, which is a bad reason
  to value expansion. A cap of ~2 chunks per doc in `fuse_rrf` is the obvious
  fix; it needs measuring the same way expansion was, and it may change the
  right value of `graph_max_slots` once it lands.
- Whether `99 topics/` notes should be retrievable directly or only reachable by
  graph expansion. They are summaries of other notes, so they may crowd out
  primaries. Partly answered: with `graph_max_slots` they can now take at most 2
  of 8 slots however they are reached, so the blast radius is bounded either
  way. Whether they *help* when they do appear is still per-question — see the
  split verdict in "Graph expansion, measured".
- `graph_discount` (0.7) and `graph_max_siblings` (5) are still shipped
  unvalidated. The measurement showed the discount is not what bounds expansion
  — the quota is — so these two now matter less than they appeared to, but
  neither has been swept.
- Whether an answer worth keeping should become a note in a `98 answers/` folder.
  That would make this a writer, and it would need the
  `obsidian-vault-writer` rules. Deliberately out of scope for v1.
