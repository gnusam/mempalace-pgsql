"""
test_db.py — Non-regression tests for PostgreSQL + pgvector storage layer.

Requires a running PostgreSQL with pgvector. Uses DATABASE_URL env var.
Run: docker compose up -d postgres && DATABASE_URL=postgresql://mempalace:mempalace@localhost:5433/mempalace pytest tests/test_db.py
"""

import os
import time
import pytest

# Skip if no DATABASE_URL
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://mempalace:mempalace@localhost:5433/mempalace"
)


@pytest.fixture(scope="module")
def db():
    from mempalace.db import PalaceDB

    db = PalaceDB(DATABASE_URL)
    # Schema already created by docker-compose init script
    # Clean test data before suite
    cur = db.conn().cursor()
    cur.execute("DELETE FROM drawers WHERE wing LIKE 'test_%'")
    cur.execute("DELETE FROM compressed WHERE id LIKE 'test_%'")
    cur.execute("DELETE FROM triples WHERE subject LIKE 'test_%' OR object LIKE 'test_%'")
    cur.execute("DELETE FROM entities WHERE id LIKE 'test_%'")
    yield db
    # Clean after
    cur = db.conn().cursor()
    cur.execute("DELETE FROM drawers WHERE wing LIKE 'test_%'")
    cur.execute("DELETE FROM compressed WHERE id LIKE 'test_%'")
    cur.execute("DELETE FROM triples WHERE subject LIKE 'test_%' OR object LIKE 'test_%'")
    cur.execute("DELETE FROM entities WHERE id LIKE 'test_%'")
    db.close()


# ── Drawer ID uniqueness (regression: diary collision) ───────────────────


class TestDrawerIDUniqueness:
    def test_diary_entries_get_unique_ids(self, db):
        """Regression: diary entries with source_file='' all got the same ID."""
        id1 = db.add_drawer("test_diary", "diary", "First entry", source_file="", chunk_index=0)
        time.sleep(0.01)  # ensure different timestamp
        id2 = db.add_drawer("test_diary", "diary", "Second entry", source_file="", chunk_index=0)
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2, "Diary entries must get unique IDs"

    def test_mcp_add_drawer_unique_ids(self, db):
        """Regression: MCP add_drawer with no source_file got duplicate IDs."""
        id1 = db.add_drawer("test_mcp", "notes", "Note about auth", source_file="")
        time.sleep(0.01)
        id2 = db.add_drawer("test_mcp", "notes", "Note about billing", source_file="")
        assert id1 != id2

    def test_mine_drawer_dedup_by_source(self, db):
        """Mine mode: same source_file + chunk_index should dedup (ON CONFLICT)."""
        id1 = db.add_drawer("test_mine", "code", "content A", source_file="/a/b.py", chunk_index=0)
        id2 = db.add_drawer("test_mine", "code", "content B", source_file="/a/b.py", chunk_index=0)
        assert id1 is not None
        assert id2 is None  # duplicate, should be skipped


# ── Filtered vector search (regression: HNSW + WHERE) ───────────────────


class TestFilteredSearch:
    def test_search_with_wing_filter(self, db):
        """Regression: HNSW index broke WHERE clause, returned empty results."""
        db.add_drawer("test_search_a", "docs", "PostgreSQL vector search with pgvector")
        db.add_drawer("test_search_b", "docs", "ChromaDB uses ONNX for embeddings")

        results = db.query("vector database search", n_results=5, where={"wing": "test_search_a"})
        assert len(results["ids"][0]) > 0, "Filtered search must return results"
        for meta in results["metadatas"][0]:
            assert meta["wing"] == "test_search_a"

    def test_search_with_compound_filter(self, db):
        """Search with $and filter on wing + room."""
        db.add_drawer("test_compound", "backend", "Flask REST API endpoint")
        db.add_drawer("test_compound", "frontend", "React component rendering")

        results = db.query(
            "API endpoint",
            n_results=5,
            where={"$and": [{"wing": "test_compound"}, {"room": "backend"}]},
        )
        assert len(results["ids"][0]) > 0
        for meta in results["metadatas"][0]:
            assert meta["room"] == "backend"

    def test_search_without_filter_uses_index(self, db):
        """Global search (no WHERE) should still work."""
        results = db.query("test query", n_results=3)
        assert "ids" in results
        assert "distances" in results


# ── Knowledge graph (regression: add_triple return value) ────────────────


class TestKnowledgeGraph:
    def test_add_triple_returns_id(self, db):
        """First add should return a triple ID."""
        tid = db.add_triple("test_max", "child_of", "test_alice", valid_from="2015-04-01")
        assert tid is not None

    def test_add_duplicate_triple_returns_existing_id(self, db):
        """Regression: duplicate triple returned None instead of existing ID."""
        tid1 = db.add_triple("test_dup_sub", "works_on", "test_dup_proj")
        tid2 = db.add_triple("test_dup_sub", "works_on", "test_dup_proj")
        assert tid1 is not None
        assert tid2 == tid1, "Duplicate triple must return existing ID, not None"

    def test_invalidate_sets_end_date(self, db):
        db.add_triple("test_inv_sub", "uses", "test_inv_tool")
        db.invalidate("test_inv_sub", "uses", "test_inv_tool", ended="2026-04-07")
        results = db.query_entity("test_inv_sub")
        ended = [r for r in results if r["predicate"] == "uses" and not r["current"]]
        assert len(ended) > 0

    def test_query_entity_both_directions(self, db):
        db.add_triple("test_dir_a", "manages", "test_dir_b")
        outgoing = db.query_entity("test_dir_a", direction="outgoing")
        incoming = db.query_entity("test_dir_b", direction="incoming")
        assert any(r["predicate"] == "manages" for r in outgoing)
        assert any(r["predicate"] == "manages" for r in incoming)

    def test_timeline(self, db):
        db.add_triple("test_tl", "started", "test_tl_proj", valid_from="2026-01-01")
        db.add_triple("test_tl", "completed", "test_tl_proj", valid_from="2026-03-15")
        tl = db.timeline("test_tl")
        assert len(tl) >= 2


# ── file_already_mined ───────────────────────────────────────────────────


class TestFileAlreadyMined:
    def test_returns_false_for_new_file(self, db):
        assert db.file_already_mined("/nonexistent/path.py") is False

    def test_returns_true_after_mining(self, db):
        db.add_drawer("test_mined", "code", "content", source_file="/test/mined.py")
        assert db.file_already_mined("/test/mined.py") is True


# ── Compressed (AAAK) ───────────────────────────────────────────────────


class TestCompressed:
    def test_upsert_compressed(self, db):
        db.upsert_compressed("test_comp_1", "TEAM:ALC|KAI|SOR", {"ratio": 8.5})
        # Second upsert should overwrite
        db.upsert_compressed("test_comp_1", "TEAM:ALC|KAI|SOR|MAY", {"ratio": 7.2})
        # No error = success


# ── Count and delete ─────────────────────────────────────────────────────


# ── Wing name auto-detection (regression: "gzyk" returned noise) ─────────


class TestAutoDetectFilter:
    def test_exact_wing_match(self, db):
        """Searching for a wing name should auto-filter to that wing."""
        db.add_drawer("test_autodetect", "code", "some rust code with cpal audio")
        db.add_drawer("test_other_wing", "code", "completely unrelated python code")

        detected = db._auto_detect_filter("test_autodetect")
        assert detected == {"wing": "test_autodetect"}

    def test_no_match_returns_none(self, db):
        """Query that doesn't match any wing/room should return None."""
        detected = db._auto_detect_filter("xyzzy_nonexistent_project")
        assert detected is None

    def test_search_by_wing_name_returns_right_wing(self, db):
        """End-to-end: searching a wing name finds content from that wing."""
        db.add_drawer("test_wingmatch", "src", "crossfade audio transition mixer")
        results = db.query("test_wingmatch", n_results=3)
        assert len(results["ids"][0]) > 0
        assert all(m["wing"] == "test_wingmatch" for m in results["metadatas"][0])

    def test_short_names_not_matched(self, db):
        """Wing names <= 2 chars should not be auto-matched from query words."""
        detected = db._auto_detect_filter("go is a language")
        # Should not match a wing named "go" if it existed
        # (we test the len > 2 guard)
        assert detected is None or len(list(detected.values())[0]) > 2


# ── Skip directories ────────────────────────────────────────────────────


class TestSkipDirs:
    def test_target_in_skip_dirs(self):
        from mempalace.miner import SKIP_DIRS

        assert "target" in SKIP_DIRS, "Rust target/ must be skipped"

    def test_vendor_in_skip_dirs(self):
        from mempalace.miner import SKIP_DIRS

        assert "vendor" in SKIP_DIRS, "PHP/Go vendor/ must be skipped"

    def test_storage_in_skip_dirs(self):
        from mempalace.miner import SKIP_DIRS

        assert "storage" in SKIP_DIRS, "Laravel storage/ must be skipped"

    def test_node_modules_in_skip_dirs(self):
        from mempalace.miner import SKIP_DIRS

        assert "node_modules" in SKIP_DIRS


# ── Count and delete ─────────────────────────────────────────────────────


class TestCountAndDelete:
    def test_count_with_filter(self, db):
        db.add_drawer("test_count", "a", "content a", source_file="/count/a.py")
        db.add_drawer("test_count", "b", "content b", source_file="/count/b.py")
        total = db.count(where={"wing": "test_count"})
        assert total >= 2

    def test_delete_drawer(self, db):
        did = db.add_drawer("test_del", "x", "to delete", source_file="/del/x.py")
        assert db.drawer_exists(did)
        db.delete_drawer(did)
        assert not db.drawer_exists(did)
