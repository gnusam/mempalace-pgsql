<div align="center">

<img src="assets/mempalace_logo.png" alt="MemPalace" width="280">

# MemPalace-PgSQL

### Fork of [MemPalace](https://github.com/milla-jovovich/mempalace) — PostgreSQL + pgvector + GPU

</div>

---

## Why this fork?

This is an experimental fork that replaces MemPalace's storage layer:

| | Upstream MemPalace | This fork |
|--|--|--|
| **Vector DB** | ChromaDB (ONNX Runtime) | PostgreSQL + pgvector |
| **Knowledge Graph** | SQLite (separate file) | Same PostgreSQL instance |
| **Embeddings** | ONNX all-MiniLM-L6-v2 (CPU) | sentence-transformers + PyTorch CUDA |
| **Deployment** | `pip install` | Docker Compose (PostgreSQL + app) |
| **GPU** | Not used | Native CUDA via PyTorch |

**Motivations:**

- **Stability** -- ChromaDB's Rust bindings segfault on Python 3.14. psycopg2 works everywhere.
- **Single database** -- Vectors and knowledge graph in one PostgreSQL instance instead of ChromaDB + SQLite side by side.
- **GPU embeddings** -- sentence-transformers with PyTorch CUDA is straightforward. No ONNX Runtime configuration needed.
- **Concurrency** -- PostgreSQL handles concurrent access natively. No more corruption from parallel mine processes.
- **Ops** -- Standard PostgreSQL tooling: `pg_dump`, replication, monitoring. Docker Compose for reproducible deployments.

Everything else stays the same: the palace structure (wings, rooms, halls, tunnels), AAAK compression, the MCP server with 19 tools, auto-save hooks.

### Performance

Benchmarked on 275k drawers across 32 projects (post-purge of minified/vendored noise), HNSW tuned for real recall (`m=32, ef_construction=200, ef_search=500`):

| Metric | This fork (pgvector) | Upstream (ChromaDB) |
|---|---|---|
| **Global search** | ~220 ms median | ~250 ms |
| **Filtered search (by wing)** | ~100 ms median | N/A (segfault on Python 3.14) |
| **Embed (GPU)** | ~6 ms/text | ~5 ms/text (CPU ONNX) |
| **Insert** | ~10 ms/drawer | ~10 ms/drawer |
| **Dedup threshold** | 0.947 cosine (more discriminant) | ~0.9 L2 |
| **Parallel mining** | 4 workers, zero DB contention | Impossible (SQLite lock) |

Cosine distance (this fork) is better suited for text embeddings than L2 (upstream). The global-search latency is higher than a naive HNSW benchmark would show because `ef_search=500` is non-default — it trades a few hundred ms for correct recall on a 250k+ corpus. The stock pgvector default (`ef_search=40`) returns top-10 results in <100 ms but with near-zero recall on out-of-distribution queries; the latency table above is the price of honest ranking.

---

## Quick Start

```bash
git clone git@github.com:gnusam/mempalace-pgsql.git
cd mempalace-pgsql

# Start PostgreSQL + pgvector
docker compose up -d postgres

# Init a project
docker compose run --rm --entrypoint python mempalace -m mempalace init /projects/myapp

# Mine
docker compose run --rm mempalace mempalace mine /projects/myapp

# Search
docker compose run --rm mempalace mempalace search "why did we switch to GraphQL"
```

Projects are mounted read-only via Docker volumes. Edit `docker-compose.yml` to add your own project paths.

### Configuration

MemPalace reads its configuration from three sources, in priority order: **environment variables**, then `~/.config/mempalace/config.json`, then built-in defaults. Inside the Compose stack `DATABASE_URL` is already wired via `docker-compose.yml` — these variables only matter if you run the code outside the container or want to override defaults.

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string (used by `mempalace.db` and every CLI/MCP entrypoint). | `postgresql://mempalace:mempalace@localhost:5433/mempalace` |
| `MEMPALACE_PALACE_PATH` | Path to the memory palace data directory (markdown mirror of the DB). `MEMPAL_PALACE_PATH` is accepted as a legacy alias. | `~/.mempalace/palace` |
| `MEMPALACE_SOURCE_DIR` | Source directory scanned by `mempalace.split_mega_files` when mining conversation transcripts. | `~/Desktop/transcripts` |

Example — running the CLI against a non-default Postgres on the host:

```bash
export DATABASE_URL=postgresql://mempalace:mempalace@db.local:5432/mempalace
export MEMPALACE_PALACE_PATH=/data/mempalace/palace
mempalace search "auth decisions"
```

Example — pointing the MCP server at a remote database:

```bash
DATABASE_URL=postgresql://user:pass@db.local:5432/mempalace \
  claude mcp add mempalace -- python -m mempalace.mcp_server
```

A persistent alternative is `~/.config/mempalace/config.json`:

```json
{
  "palace_path": "/data/mempalace/palace",
  "database_url": "postgresql://mempalace:mempalace@db.local:5432/mempalace"
}
```

Environment variables always win over the JSON file.

### GPU

The app container uses `nvidia/cuda` with PyTorch CUDA. If you have an NVIDIA GPU with the container toolkit installed, embeddings run on GPU automatically. No configuration needed -- sentence-transformers detects CUDA and uses it.

```
$ docker compose run --rm --entrypoint python mempalace -c "import torch; print(torch.cuda.get_device_name(0))"
NVIDIA GeForce RTX 4070 Laptop GPU
```

### MCP Server (Claude Code)

```bash
claude mcp add mempalace -- docker compose -f /path/to/docker-compose.yml run --rm -i mempalace mempalace.mcp_server
```

19 tools available: search, browse, add/delete drawers, knowledge graph, agent diary. Same as upstream.

**Note:** The MCP server is a long-running stdio process. Code changes require restarting Claude Code (or the MCP server) to take effect -- the server does not hot-reload.

---

## Architecture

```
docker-compose.yml
├── postgres (pgvector/pgvector:pg16)
│   ├── drawers        -- text + vector(384) + metadata (JSONB)
│   ├── compressed     -- AAAK compressed versions
│   ├── entities       -- knowledge graph nodes
│   └── triples        -- knowledge graph edges (temporal)
│
└── mempalace (nvidia/cuda:12.9 + Python 3.12)
    ├── sentence-transformers (all-MiniLM-L6-v2, CUDA)
    ├── psycopg2 + pgvector
    └── MCP server (stdio)
```

Vector search uses pgvector's HNSW index with cosine distance for global queries. When a query matches a wing or room name, the search auto-filters to that scope and falls back to sequential scan (HNSW doesn't support pre-filtering). The embedding model (`all-MiniLM-L6-v2`, 384 dimensions) is the same as upstream MemPalace.

**HNSW tuning.** The index is built with `WITH (m = 32, ef_construction = 200)` (vs pgvector defaults 16/64) and queries run with `SET LOCAL hnsw.ef_search = 500` (vs default 40). These non-default values are necessary for recall on corpora over ~100 k drawers — the stock pgvector defaults produced ~0% recall@10 on a 400 k drawer palace, returning noise clusters while the true top-k stayed unreachable. Postgres itself is started with `maintenance_work_mem=4GB` + `shm_size=2GB` so REINDEX can hold the full HNSW graph in memory and run parallel maintenance workers; building under the default 64 MB silently degrades to an on-disk build with severely worse recall (watch for `NOTICE: hnsw graph no longer fits into maintenance_work_mem after N tuples` in the REINDEX output — if you see it, the build is bad).

Metadata queries (`list_wings`, `list_rooms`, `get_taxonomy`, `status`) use direct `GROUP BY` SQL rather than fetching rows -- instant even on 400k+ drawers.

### Mining hygiene

Mining respects `.gitignore` by default (nested `.gitignore` files, anchors, negations, and directory-only rules are all honoured). On top of that, a baseline skip list excludes common noise:

```
.git  node_modules  __pycache__  .venv  venv  env  dist  build  .next
coverage  .mempalace  .ruff_cache  .mypy_cache  .pytest_cache  .cache
.tox  .nox  .idea  .vscode  .ipynb_checkpoints  .eggs  htmlcov
target  vendor  .gradle  storage
```

Additional anti-noise filters applied to each file:
- **Minified bundles** skipped by filename pattern: `*.min.js`, `*.min.css`, `*.min.mjs`, `*-bundle.js`, `*.bundle.js`, `*.umd.js`, `*.esm.js`, `*.prod.js`.
- **Machine-generated files** skipped by average line length: any file whose `len(content) / line_count > 400` is dropped. Hand-written code rarely exceeds ~200 chars/line; minified JS/CSS and dumped JSON blobs (e.g. Symfony Intl CLDR data) hit thousands.
- **Lockfiles** skipped by name: `package-lock.json`, `yarn.lock`, `composer.lock`, `Cargo.lock`, `poetry.lock`, `pnpm-lock.yaml`.

These were added after a diagnostic session revealed that minified Swagger assets and vendored CLDR JSON had polluted ~32% of the palace (130 k of 405 k drawers) and were drowning real documentation in the semantic-search top-k.

Override at mine time:

```bash
mempalace mine ./project --no-gitignore                # disable gitignore matching
mempalace mine ./project --include-ignored docs,build  # force-include otherwise-ignored paths
```

---

## Parallel Mining

PostgreSQL handles concurrent writes natively. You can mine multiple projects simultaneously:

```bash
# Launch 4 workers in parallel
for project in app1 app2 app3 app4; do
  docker compose run --rm -d --name mp-$project mempalace mempalace mine /projects/$project
done
```

No corruption, no locking issues. ChromaDB's SQLite backend couldn't do this.

---

## Development

Source code is mounted into the container via `docker-compose.yml` -- no rebuild needed when editing code.

```bash
# Run tests (requires PostgreSQL running)
docker compose up -d postgres
docker compose run --rm --entrypoint bash mempalace -c "pip install pytest -q && pytest tests/ -v"
```

### Test coverage

69 tests, combining the expanded upstream suite with PostgreSQL-specific regressions:

| Suite | Count | What |
|---|---|---|
| AAAK dialect | 16 | Entity codes, emotions, topics, key sentence extraction, zettel encoding, decode roundtrip, honest token counting |
| Scan / .gitignore | 11 | Nested .gitignore, anchors, negations, dir-only rules, `--no-gitignore`, `--include-ignored` overrides |
| PostgreSQL storage (`test_db.py`) | ~20 | Drawer ID uniqueness, HNSW filtered search, wing auto-detect, KG triples (add/query/invalidate/timeline), CRUD, compressed upsert |
| Split mega-files | 5 | Known-people config, name detection, fallback behaviour |
| Config | 4 | Defaults, file config, env override, init |
| Normalize | 3 | Plain text, Claude JSON, empty input |
| Version consistency | 2 | Package version matches `pyproject.toml` and MCP `initialize` |
| Project mining | 1 | End-to-end mine with `mempalace.yaml` |
| Conversation mining | 1 | End-to-end convo mine + search |

Backend-agnostic tests (56) run without PostgreSQL. The 13 DB-touching tests require the Compose stack: `docker compose up -d postgres`.

---

## Upstream MemPalace

For the full documentation on the palace concept, AAAK dialect, memory layers, benchmarks, and mining modes, see the upstream project:

**[github.com/milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace)**

**Sync status:** this fork's `main` rebases a single squash commit on top of upstream `main` at [`71736a3`](https://github.com/milla-jovovich/mempalace/commit/71736a3) (PR #142, packaging cleanup), audited against upstream through [`2981433`](https://github.com/milla-jovovich/mempalace/commit/2981433) (v3.1.0 + PR #315). Ported from that window: the `--palace` CLI flag for `mempalace.mcp_server` (upstream [PR #264](https://github.com/milla-jovovich/mempalace/pull/264), adapted — our KG lives in Postgres so the KG-path portion is dropped), `pytest-cov` coverage reporting in CI with a 30% floor (upstream `9de302f`), the mine summary stats fix (upstream [`75eb7ff`](https://github.com/milla-jovovich/mempalace/commit/75eb7ff), PR #165 by @adv3nt3), deterministic content IDs + `ON CONFLICT DO UPDATE` upsert to fix data stagnation (upstream [`a4149ab`](https://github.com/milla-jovovich/mempalace/commit/a4149ab), PR #140 findings #6/#11/#13 by @igorls), mtime-aware `file_already_mined` so modified files actually re-mine (upstream [`bf88daa`](https://github.com/milla-jovovich/mempalace/commit/bf88daa), same PR), MCP null-args hang fix so `"arguments": null` no longer crashes the server (upstream [`0720fb8`](https://github.com/milla-jovovich/mempalace/commit/0720fb8), PR #399 by @bensig, fixes #394), MCP protocol version negotiation so newer Claude Code clients can connect (upstream [`950d52b`](https://github.com/milla-jovovich/mempalace/commit/950d52b), PR #324 by @virgil-at-biocompute), 500 MB file-size guard on `normalize.py` and `split_mega_files.py` (upstream [`0720fb8`](https://github.com/milla-jovovich/mempalace/commit/0720fb8), PR #399, fixes #396), the `mempalace mcp` setup-guidance subcommand (upstream [`2981433`](https://github.com/milla-jovovich/mempalace/commit/2981433), PR #315 by @kpulik, adapted to print the Docker Compose invocation instead of `python -m`), and a 10 MB per-file size ceiling + symlink skip during `scan_project` / `scan_convos` (cherry-picked from upstream [`1d19dfc`](https://github.com/milla-jovovich/mempalace/commit/1d19dfc), PR #252 by @anthonyonazure / @bensig — only the two scan-time safety filters; the rest of that commit is not applicable, see below). All other portable fixes merged into upstream since the fork point are absorbed: security hardening (shell injection, path traversal, hook security, error sanitisation), bounded queries, `.gitignore`-aware mining, MCP integer coercion, Windows Unicode fix, AAAK honest stats, and the expanded upstream test suite. Intentionally skipped because they don't apply to the PostgreSQL backend or this fork's delivery model: SQLite/ChromaDB-specific fixes (WAL mode, batched unbounded reads, `chromadb` version pin, ChromaDB client cache [`a67b00d`](https://github.com/milla-jovovich/mempalace/commit/a67b00d) — already achieved here via `get_db()`, ChromaDB telemetry/CoreML silencing `df33550`, Windows ChromaDB handle release [`58b8d5b`](https://github.com/milla-jovovich/mempalace/commit/58b8d5b)), the KG co-location portion of PR #264 and the security rework of `knowledge_graph.py` inside PR #252 (our KG lives in Postgres), the `palace.py` consolidation introduced in PR #252 (we don't share that module), upstream's `cmd_repair` infinite-recursion fix (ChromaDB-specific), upstream's 94-test ChromaDB-stress benchmark suite (PR #223 — targets ChromaDB-specific OOM and degradation behaviours we don't hit), upstream test-file ruff-format commits (`a0bcd0c`, `af42a85` — touch files we don't have), upstream's `parse_known_args` pytest-collection fix (`edf8f36` — already absorbed in fork commit `a309d16` when we ported PR #264), upstream's macOS/Windows CI split ([`43c5a47`](https://github.com/milla-jovovich/mempalace/commit/43c5a47), Windows mtime test fix [`1c48f4d`](https://github.com/milla-jovovich/mempalace/commit/1c48f4d) — we're Linux-only), upstream test coverage commits that touch `test_knowledge_graph.py` / `test_mcp_server.py` / `test_searcher.py` (dropped at PG migration in `abbab4f`), the md5→sha256 drawer-ID change inside PR #252 (would invalidate all existing drawer IDs without real benefit), the metadata 30s TTL cache inside PR #252 (upstream itself removed it in [`c2308a1`](https://github.com/milla-jovovich/mempalace/commit/c2308a1) for breaking test isolation), the `hooks/mempal_save_hook.sh` shell-eval sanitisation inside PR #252 (our hook uses command substitution into variables, not `eval` — so there is no injection path to fix), the v3.0.x / v3.1.0 version bump chore commits, and upstream's Claude Code plugin / marketplace ecosystem (`.claude-plugin/`, `.codex-plugin/`, `hooks_cli.py`) — this fork ships via Docker Compose, not the pip-install marketplace flow.

Changes on top of upstream are limited to the storage backend, with improvements to search (wing auto-detection, cosine distance) and ops (Docker Compose, parallel workers).

**Fork-specific MCP reliability fixes (2026-04-09):**

- `db.query` now restores `autocommit` and rolls back in a `try/finally`, so a failing filtered search no longer leaves the connection in an aborted-transaction state that poisons every subsequent `check_duplicate` / `search` call (previously observed as `mempalace_check_duplicate` returning KO after the first slow query).
- `db.query` gained an `auto_detect` flag; `mempalace_check_duplicate` now calls it with `auto_detect=False` so a room/wing name appearing inside the content being checked no longer scopes the duplicate search to that wing and no longer forces a full sequential scan — this was the root cause of the `mempalace_add_drawer` connection drop on multi-KB payloads, since `add_drawer` calls `check_duplicate` internally and long content was much more likely to match a room name.
- `_auto_detect_filter` skips NULL/empty wings and rooms instead of crashing on `None.lower()`.
- The MCP handler surfaces the actual exception type and message (bounded to 500 chars) instead of a generic "Internal tool error", and calls `db.reset()` on `OperationalError` / `InterfaceError` / `InFailedSqlTransaction` so the next tool call gets a fresh connection. Two regression tests were added in `tests/test_db.py::TestAutoDetectFilter`.

---

## How You Actually Use It

After the one-time setup (install → init → mine), you don't run MemPalace commands manually. Your AI uses it for you. There are two ways, depending on which AI you use.

### With Claude, ChatGPT, Cursor (MCP-compatible tools)

```bash
# Connect MemPalace once
claude mcp add mempalace -- python -m mempalace.mcp_server
```

Now your AI has 19 tools available through MCP. Ask it anything:

> *"What did we decide about auth last month?"*

Claude calls `mempalace_search` automatically, gets verbatim results, and answers you. You never type `mempalace search` again. The AI handles it.

### With local models (Llama, Mistral, or any offline LLM)

Local models generally don't speak MCP yet. Two approaches:

**1. Wake-up command** — load your world into the model's context:

```bash
mempalace wake-up > context.txt
# Paste context.txt into your local model's system prompt
```

This gives your local model ~170 tokens of critical facts (in AAAK if you prefer) before you ask a single question.

**2. CLI search** — query on demand, feed results into your prompt:

```bash
mempalace search "auth decisions" > results.txt
# Include results.txt in your prompt
```

Or use the Python API:

```python
from mempalace.searcher import search_memories
results = search_memories("auth decisions", palace_path="~/.mempalace/palace")
# Inject into your local model's context
```

Either way — your entire memory stack runs offline. PostgreSQL on your machine, Llama on your machine, AAAK for compression, zero cloud calls.

---

## The Problem

Decisions happen in conversations now. Not in docs. Not in Jira. In conversations with Claude, ChatGPT, Copilot. The reasoning, the tradeoffs, the "we tried X and it failed because Y" — all trapped in chat windows that evaporate when the session ends.

**Six months of daily AI use = 19.5 million tokens.** That's every decision, every debugging session, every architecture debate. Gone.

| Approach | Tokens loaded | Annual cost |
|----------|--------------|-------------|
| Paste everything | 19.5M — doesn't fit any context window | Impossible |
| LLM summaries | ~650K | ~$507/yr |
| **MemPalace wake-up** | **~170 tokens** | **~$0.70/yr** |
| **MemPalace + 5 searches** | **~13,500 tokens** | **~$10/yr** |

MemPalace loads 170 tokens of critical facts on wake-up — your team, your projects, your preferences. Then searches only when needed. $10/year to remember everything vs $507/year for summaries that lose context.

---

## How It Works

### The Palace

The layout is fairly simple, though it took a long time to get there.

It starts with a **wing**. Every project, person, or topic you're filing gets its own wing in the palace.

Each wing has **rooms** connected to it, where information is divided into subjects that relate to that wing — so every room is a different element of what your project contains. Project ideas could be one room, employees could be another, financial statements another. There can be an endless number of rooms that split the wing into sections. The MemPalace install detects these for you automatically, and of course you can personalize it any way you feel is right.

Every room has a **closet** connected to it, and here's where things get interesting. We've developed an AI language called **AAAK**. Don't ask — it's a whole story of its own. Your agent learns the AAAK shorthand every time it wakes up. Because AAAK is essentially English, but a very truncated version, your agent understands how to use it in seconds. It comes as part of the install, built into the MemPalace code. In our next update, we'll add AAAK directly to the closets, which will be a real game changer — the amount of info in the closets will be much bigger, but it will take up far less space and far less reading time for your agent.

Inside those closets are **drawers**, and those drawers are where your original files live. In this first version, we haven't used AAAK as a closet tool, but even so, the summaries have shown **96.6% recall** in all the benchmarks we've done across multiple benchmarking platforms. Once the closets use AAAK, searches will be even faster while keeping every word exact. But even now, the closet approach has been a huge boon to how much info is stored in a small space — it's used to easily point your AI agent to the drawer where your original file lives. You never lose anything, and all this happens in seconds.

There are also **halls**, which connect rooms within a wing, and **tunnels**, which connect rooms from different wings to one another. So finding things becomes truly effortless — we've given the AI a clean and organized way to know where to start searching, without having to look through every keyword in huge folders.

You say what you're looking for and boom, it already knows which wing to go to. Just *that* in itself would have made a big difference. But this is beautiful, elegant, organic, and most importantly, efficient.

```
  ┌─────────────────────────────────────────────────────────────┐
  │  WING: Person                                              │
  │                                                            │
  │    ┌──────────┐  ──hall──  ┌──────────┐                    │
  │    │  Room A  │            │  Room B  │                    │
  │    └────┬─────┘            └──────────┘                    │
  │         │                                                  │
  │         ▼                                                  │
  │    ┌──────────┐      ┌──────────┐                          │
  │    │  Closet  │ ───▶ │  Drawer  │                          │
  │    └──────────┘      └──────────┘                          │
  └─────────┼──────────────────────────────────────────────────┘
            │
          tunnel
            │
  ┌─────────┼──────────────────────────────────────────────────┐
  │  WING: Project                                             │
  │         │                                                  │
  │    ┌────┴─────┐  ──hall──  ┌──────────┐                    │
  │    │  Room A  │            │  Room C  │                    │
  │    └────┬─────┘            └──────────┘                    │
  │         │                                                  │
  │         ▼                                                  │
  │    ┌──────────┐      ┌──────────┐                          │
  │    │  Closet  │ ───▶ │  Drawer  │                          │
  │    └──────────┘      └──────────┘                          │
  └─────────────────────────────────────────────────────────────┘
```

**Wings** — a person or project. As many as you need.
**Rooms** — specific topics within a wing. Auth, billing, deploy — endless rooms.
**Halls** — connections between related rooms *within* the same wing. If Room A (auth) and Room B (security) are related, a hall links them.
**Tunnels** — connections *between* wings. When Person A and a Project both have a room about "auth," a tunnel cross-references them automatically.
**Closets** — compressed summaries that point to the original content. Fast for AI to read.
**Drawers** — the original verbatim files. The exact words, never summarized.

**Halls** are memory types — the same in every wing, acting as corridors:
- `hall_facts` — decisions made, choices locked in
- `hall_events` — sessions, milestones, debugging
- `hall_discoveries` — breakthroughs, new insights
- `hall_preferences` — habits, likes, opinions
- `hall_advice` — recommendations and solutions

**Rooms** are named ideas — `auth-migration`, `graphql-switch`, `ci-pipeline`. When the same room appears in different wings, it creates a **tunnel** — connecting the same topic across domains:

```
wing_kai       / hall_events / auth-migration  → "Kai debugged the OAuth token refresh"
wing_driftwood / hall_facts  / auth-migration  → "team decided to migrate auth to Clerk"
wing_priya     / hall_advice / auth-migration  → "Priya approved Clerk over Auth0"
```

Same room. Three wings. The tunnel connects them.

### Why Structure Matters

Tested on 22,000+ real conversation memories:

```
Search all closets:          60.9%  R@10
Search within wing:          73.1%  (+12%)
Search wing + hall:          84.8%  (+24%)
Search wing + room:          94.8%  (+34%)
```

Wings and rooms aren't cosmetic. They're a **34% retrieval improvement**. The palace structure is the product.

### The Memory Stack

| Layer | What | Size | When |
|-------|------|------|------|
| **L0** | Identity — who is this AI? | ~50 tokens | Always loaded |
| **L1** | Critical facts — team, projects, preferences | ~120 tokens (AAAK) | Always loaded |
| **L2** | Room recall — recent sessions, current project | On demand | When topic comes up |
| **L3** | Deep search — semantic query across all closets | On demand | When explicitly asked |

Your AI wakes up with L0 + L1 (~170 tokens) and knows your world. Searches only fire when needed.

### AAAK Dialect

AAAK is a structured symbolic summary format — lossy, but readable by any LLM without a decoder. It extracts entities, topics, a key sentence, emotions, and flags into a compact representation. It works with **Claude, GPT, Gemini, Llama, Mistral** — any model that reads text. Run it against a local Llama model and your whole memory stack stays offline.

Note: AAAK is *lossy summarisation*, not compression — the original text cannot be reconstructed from AAAK output. Upstream [PR #147](https://github.com/milla-jovovich/mempalace/pull/147) corrected earlier "30x lossless compression" claims and introduced a word-based token estimator; raw verbatim mode (drawers) is what delivers the published 96.6% LongMemEval score.

**English (~1000 tokens):**
```
Priya manages the Driftwood team: Kai (backend, 3 years), Soren (frontend),
Maya (infrastructure), and Leo (junior, started last month). They're building
a SaaS analytics platform. Current sprint: auth migration to Clerk.
Kai recommended Clerk over Auth0 based on pricing and DX.
```

**AAAK (~120 tokens):**
```
TEAM: PRI(lead) | KAI(backend,3yr) SOR(frontend) MAY(infra) LEO(junior,new)
PROJ: DRIFTWOOD(saas.analytics) | SPRINT: auth.migration→clerk
DECISION: KAI.rec:clerk>auth0(pricing+dx) | ★★★★
```

Same information. 8x fewer tokens. Your AI learns AAAK automatically from the MCP server — no manual setup.

### Contradiction Detection

MemPalace catches mistakes before they reach you:

```
Input:  "Soren finished the auth migration"
Output: 🔴 AUTH-MIGRATION: attribution conflict — Maya was assigned, not Soren

Input:  "Kai has been here 2 years"
Output: 🟡 KAI: wrong_tenure — records show 3 years (started 2023-04)

Input:  "The sprint ends Friday"
Output: 🟡 SPRINT: stale_date — current sprint ends Thursday (updated 2 days ago)
```

Facts checked against the knowledge graph. Ages, dates, and tenures calculated dynamically — not hardcoded.

---

## Real-World Examples

### Solo developer across multiple projects

```bash
# Mine each project's conversations
mempalace mine ~/chats/orion/  --mode convos --wing orion
mempalace mine ~/chats/nova/   --mode convos --wing nova
mempalace mine ~/chats/helios/ --mode convos --wing helios

# Six months later: "why did I use Postgres here?"
mempalace search "database decision" --wing orion
# → "Chose Postgres over SQLite because Orion needs concurrent writes
#    and the dataset will exceed 10GB. Decided 2025-11-03."

# Cross-project search
mempalace search "rate limiting approach"
# → finds your approach in Orion AND Nova, shows the differences
```

### Team lead managing a product

```bash
# Mine Slack exports and AI conversations
mempalace mine ~/exports/slack/ --mode convos --wing driftwood
mempalace mine ~/.claude/projects/ --mode convos

# "What did Soren work on last sprint?"
mempalace search "Soren sprint" --wing driftwood
# → 14 closets: OAuth refactor, dark mode, component library migration

# "Who decided to use Clerk?"
mempalace search "Clerk decision" --wing driftwood
# → "Kai recommended Clerk over Auth0 — pricing + developer experience.
#    Team agreed 2026-01-15. Maya handling the migration."
```

### Before mining: split mega-files

Some transcript exports concatenate multiple sessions into one huge file:

```bash
mempalace split ~/chats/                      # split into per-session files
mempalace split ~/chats/ --dry-run            # preview first
mempalace split ~/chats/ --min-sessions 3     # only split files with 3+ sessions
```

---

## Knowledge Graph

Temporal entity-relationship triples — like Zep's Graphiti, but PostgreSQL instead of Neo4j. Local and free.

```python
from mempalace.db import get_db

db = get_db()
db.add_triple("Kai", "works_on", "Orion", valid_from="2025-06-01")
db.add_triple("Maya", "assigned_to", "auth-migration", valid_from="2026-01-15")
db.add_triple("Maya", "completed", "auth-migration", valid_from="2026-02-01")

# What's Kai working on?
db.query_entity("Kai")
# → [Kai → works_on → Orion (current), Kai → recommended → Clerk (2026-01)]

# What was true in January?
db.query_entity("Maya", as_of="2026-01-20")
# → [Maya → assigned_to → auth-migration (active)]

# Timeline
db.timeline("Orion")
# → chronological story of the project
```

Facts have validity windows. When something stops being true, invalidate it:

```python
db.invalidate("Kai", "works_on", "Orion", ended="2026-03-01")
```

Now queries for Kai's current work won't return Orion. Historical queries still will.

| Feature | MemPalace | Zep (Graphiti) |
|---------|-----------|----------------|
| Storage | PostgreSQL (local) | Neo4j (cloud) |
| Cost | Free | $25/mo+ |
| Temporal validity | Yes | Yes |
| Self-hosted | Always | Enterprise only |
| Privacy | Everything local | SOC 2, HIPAA |

---

## Specialist Agents

Create agents that focus on specific areas. Each agent gets its own wing and diary in the palace — not in your CLAUDE.md. Add 50 agents, your config stays the same size.

```
~/.mempalace/agents/
  ├── reviewer.json       # code quality, patterns, bugs
  ├── architect.json      # design decisions, tradeoffs
  └── ops.json            # deploys, incidents, infra
```

Your CLAUDE.md just needs one line:

```
You have MemPalace agents. Run mempalace_list_agents to see them.
```

The AI discovers its agents from the palace at runtime. Each agent:

- **Has a focus** — what it pays attention to
- **Keeps a diary** — written in AAAK, persists across sessions
- **Builds expertise** — reads its own history to stay sharp in its domain

```
# Agent writes to its diary after a code review
mempalace_diary_write("reviewer",
    "PR#42|auth.bypass.found|missing.middleware.check|pattern:3rd.time.this.quarter|★★★★")

# Agent reads back its history
mempalace_diary_read("reviewer", last_n=10)
# → last 10 findings, compressed in AAAK
```

Each agent is a specialist lens on your data. The reviewer remembers every bug pattern it's seen. The architect remembers every design decision. The ops agent remembers every incident. They don't share a scratchpad — they each maintain their own memory.

Letta charges $20–200/mo for agent-managed memory. MemPalace does it with a wing.

---

## MCP Server

```bash
claude mcp add mempalace -- docker compose -f /path/to/docker-compose.yml run --rm -i mempalace mempalace.mcp_server
```

### 19 Tools

**Palace (read)**

| Tool | What |
|------|------|
| `mempalace_status` | Palace overview + AAAK spec + memory protocol |
| `mempalace_list_wings` | Wings with counts |
| `mempalace_list_rooms` | Rooms within a wing |
| `mempalace_get_taxonomy` | Full wing → room → count tree |
| `mempalace_search` | Semantic search with wing/room filters |
| `mempalace_check_duplicate` | Check before filing |
| `mempalace_get_aaak_spec` | AAAK dialect reference |

**Palace (write)**

| Tool | What |
|------|------|
| `mempalace_add_drawer` | File verbatim content |
| `mempalace_delete_drawer` | Remove by ID |

**Knowledge Graph**

| Tool | What |
|------|------|
| `mempalace_kg_query` | Entity relationships with time filtering |
| `mempalace_kg_add` | Add facts |
| `mempalace_kg_invalidate` | Mark facts as ended |
| `mempalace_kg_timeline` | Chronological entity story |
| `mempalace_kg_stats` | Graph overview |

**Navigation**

| Tool | What |
|------|------|
| `mempalace_traverse` | Walk the graph from a room across wings |
| `mempalace_find_tunnels` | Find rooms bridging two wings |
| `mempalace_graph_stats` | Graph connectivity overview |

**Agent Diary**

| Tool | What |
|------|------|
| `mempalace_diary_write` | Write AAAK diary entry |
| `mempalace_diary_read` | Read recent diary entries |

The AI learns AAAK and the memory protocol automatically from the `mempalace_status` response. No manual configuration.

---

## Auto-Save Hooks

Two hooks for Claude Code that automatically save memories during work:

**Save Hook** — every 15 messages, triggers a structured save. Topics, decisions, quotes, code changes. Also regenerates the critical facts layer.

**PreCompact Hook** — fires before context compression. Emergency save before the window shrinks.

```json
{
  "hooks": {
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/mempalace/hooks/mempal_save_hook.sh"}]}],
    "PreCompact": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/mempalace/hooks/mempal_precompact_hook.sh"}]}]
  }
}
```

---

## Benchmarks

Tested on standard academic benchmarks — reproducible, published datasets.

| Benchmark | Mode | Score | API Calls |
|-----------|------|-------|-----------|
| **LongMemEval R@5** | Raw (vector search only) | **96.6%** | Zero |
| **LongMemEval R@5** | Hybrid + Haiku rerank | **100%** (500/500) | ~500 |
| **LoCoMo R@10** | Raw, session level | **60.3%** | Zero |
| **Personal palace R@10** | Heuristic bench | **85%** | Zero |
| **Palace structure impact** | Wing+room filtering | **+34%** R@10 | Zero |

The 96.6% raw score is the highest published LongMemEval result requiring no API key, no cloud, and no LLM at any stage.

### vs Published Systems

| System | LongMemEval R@5 | API Required | Cost |
|--------|----------------|--------------|------|
| **MemPalace (hybrid)** | **100%** | Optional | Free |
| Supermemory ASMR | ~99% | Yes | — |
| **MemPalace (raw)** | **96.6%** | **None** | **Free** |
| Mastra | 94.87% | Yes (GPT) | API costs |
| Mem0 | ~85% | Yes | $19–249/mo |
| Zep | ~85% | Yes | $25/mo+ |

---

## All Commands

```bash
# Setup
mempalace init <dir>                              # guided onboarding + AAAK bootstrap

# Mining
mempalace mine <dir>                              # mine project files
mempalace mine <dir> --mode convos                # mine conversation exports
mempalace mine <dir> --mode convos --wing myapp   # tag with a wing name

# Splitting
mempalace split <dir>                             # split concatenated transcripts
mempalace split <dir> --dry-run                   # preview

# Search
mempalace search "query"                          # search everything
mempalace search "query" --wing myapp             # within a wing
mempalace search "query" --room auth-migration    # within a room

# Memory stack
mempalace wake-up                                 # load L0 + L1 context
mempalace wake-up --wing driftwood                # project-specific

# Compression
mempalace compress --wing myapp                   # AAAK compress

# Status
mempalace status                                  # palace overview
```

All commands accept `--palace <path>` to override the default location.

---

## Configuration

### Global (`~/.mempalace/config.json`)

```json
{
  "palace_path": "/custom/path/to/palace",
  "collection_name": "mempalace_drawers",
  "people_map": {"Kai": "KAI", "Priya": "PRI"}
}
```

### Wing config (`~/.mempalace/wing_config.json`)

Generated by `mempalace init`. Maps your people and projects to wings:

```json
{
  "default_wing": "wing_general",
  "wings": {
    "wing_kai": {"type": "person", "keywords": ["kai", "kai's"]},
    "wing_driftwood": {"type": "project", "keywords": ["driftwood", "analytics", "saas"]}
  }
}
```

### Identity (`~/.mempalace/identity.txt`)

Plain text. Becomes Layer 0 — loaded every session.

---

## File Reference

| File | What |
|------|------|
| `cli.py` | CLI entry point |
| `config.py` | Configuration loading and defaults |
| `normalize.py` | Converts 5 chat formats to standard transcript |
| `mcp_server.py` | MCP server — 19 tools, AAAK auto-teach, memory protocol |
| `miner.py` | Project file ingest |
| `convo_miner.py` | Conversation ingest — chunks by exchange pair |
| `db.py` | PostgreSQL + pgvector storage layer (replaces ChromaDB + SQLite, hosts the temporal knowledge graph) |
| `init_schema.sql` | Database schema (drawers, compressed, entities, triples) |
| `searcher.py` | Semantic search via pgvector |
| `layers.py` | 4-layer memory stack |
| `dialect.py` | AAAK dialect — structured symbolic summary format |
| `palace_graph.py` | Room-based navigation graph |
| `onboarding.py` | Guided setup — generates AAAK bootstrap + wing config |
| `entity_registry.py` | Entity code registry |
| `entity_detector.py` | Auto-detect people and projects from content |
| `split_mega_files.py` | Split concatenated transcripts into per-session files |
| `hooks/mempal_save_hook.sh` | Auto-save every N messages |
| `hooks/mempal_precompact_hook.sh` | Emergency save before compaction |

---

## Project Structure

```
mempalace/
├── README.md                  ← you are here
├── mempalace/                 ← core package (README)
│   ├── cli.py                 ← CLI entry point
│   ├── mcp_server.py          ← MCP server (19 tools)
│   ├── db.py                  ← PostgreSQL + pgvector storage + temporal KG
│   ├── init_schema.sql        ← database schema
│   ├── palace_graph.py        ← room navigation graph
│   ├── dialect.py             ← AAAK compression
│   ├── miner.py               ← project file ingest
│   ├── convo_miner.py         ← conversation ingest
│   ├── searcher.py            ← semantic search
│   ├── onboarding.py          ← guided setup
│   └── ...                    ← see mempalace/README.md
├── benchmarks/                ← reproducible benchmark runners
│   ├── README.md              ← reproduction guide
│   ├── BENCHMARKS.md          ← full results + methodology
│   ├── longmemeval_bench.py   ← LongMemEval runner
│   ├── locomo_bench.py        ← LoCoMo runner
│   └── membench_bench.py      ← MemBench runner
├── hooks/                     ← Claude Code auto-save hooks
│   ├── README.md              ← hook setup guide
│   ├── mempal_save_hook.sh    ← save every N messages
│   └── mempal_precompact_hook.sh ← emergency save
├── examples/                  ← usage examples
│   ├── basic_mining.py
│   ├── convo_import.py
│   └── mcp_setup.md
├── tests/                     ← test suite (README)
├── assets/                    ← logo + brand assets
└── pyproject.toml             ← package config (v3.0.0)
```

---

## Requirements

- Docker + Docker Compose
- NVIDIA GPU + container toolkit (optional, for GPU-accelerated embeddings)

No API key. No internet after install. Everything local.

```bash
docker compose up -d
```

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

## License

MIT — see [LICENSE](LICENSE).

<!-- Link Definitions -->
[version-shield]: https://img.shields.io/badge/version-3.0.0--pgsql-4dc9f6?style=flat-square&labelColor=0a0e14
[release-link]: https://github.com/gnusam/mempalace-pgsql/releases
[python-shield]: https://img.shields.io/badge/python-3.12-7dd8f8?style=flat-square&labelColor=0a0e14&logo=python&logoColor=7dd8f8
[python-link]: https://www.python.org/
[license-shield]: https://img.shields.io/badge/license-MIT-b0e8ff?style=flat-square&labelColor=0a0e14
[license-link]: https://github.com/gnusam/mempalace-pgsql/blob/main/LICENSE
[discord-shield]: https://img.shields.io/badge/discord-join-5865F2?style=flat-square&labelColor=0a0e14&logo=discord&logoColor=5865F2
[discord-link]: https://discord.com/invite/ycTQQCu6kn
