"""
Tests for the noise-reduction changes ported from upstream 6aebf45 (PR
adaptation — i18n portion dropped because this fork has no i18n layer).

Focus: behavioural assertions on the public surface (extract_candidates,
score_entity, classify_entity, scan_for_detection) — not regex internals.
"""

from pathlib import Path

from mempalace.entity_detector import (
    SKIP_FILENAMES,
    classify_entity,
    extract_candidates,
    score_entity,
    scan_for_detection,
)


# --- CamelCase extraction ---


def test_extract_candidates_keeps_camelcase_whole():
    text = " ".join(["MemPalace ChromaDB OpenAI"] * 4)
    candidates = extract_candidates(text)
    assert "MemPalace" in candidates
    assert "ChromaDB" in candidates
    assert "OpenAI" in candidates
    # The CamelCase fragments must NOT be re-counted as separate single words
    assert "Mem" not in candidates
    assert "Chroma" not in candidates


# --- Versioned regex no longer matches arbitrary hyphenated compounds ---


def test_versioned_signal_only_fires_on_actual_versions():
    text = " ".join(["MemPal-v1 used MemPal-v1 again with MemPal-v1"] * 2)
    scores_versioned = score_entity("MemPal", text, text.splitlines())
    text_compound = " ".join(["context-manager and context-manager and context-manager"] * 2)
    scores_compound = score_entity("context", text_compound, text_compound.splitlines())
    # Versioned variant scores as a project signal
    versioned_signals = " ".join(scores_versioned["project_signals"])
    compound_signals = " ".join(scores_compound["project_signals"])
    assert "versioned" in versioned_signals
    assert "versioned" not in compound_signals


# --- Bare-colon dialogue requires at least 2 hits ---


def test_dialogue_bare_colon_requires_two_matches():
    # Single metadata line — must NOT count as dialogue
    single = "Created: 2026-04-21\nSome other content here.\n"
    scores_single = score_entity("Created", single, single.splitlines())
    assert all("dialogue" not in s for s in scores_single["person_signals"])

    # Two NAME: lines — counts as dialogue
    repeated = "Sam: hey there\nSomething happens.\nSam: still here\n"
    scores_repeated = score_entity("Sam", repeated, repeated.splitlines())
    dialogue_signals = [s for s in scores_repeated["person_signals"] if "dialogue" in s]
    assert dialogue_signals


# --- Strong pronoun signal still classifies as person ---


def test_classify_strong_pronoun_signal_keeps_person():
    # 16 pronoun hits across 30 mentions = 0.53 ratio, well above 0.2 floor
    scores = {
        "person_score": 32,  # 16 pronoun hits * 2
        "project_score": 0,
        "person_signals": ["pronoun nearby (16x)"],
        "project_signals": [],
    }
    result = classify_entity("Lu", frequency=30, scores=scores)
    assert result["type"] == "person"


def test_classify_weak_pronoun_signal_remains_uncertain():
    # Only 3 pronoun hits — below the >=5 floor
    scores = {
        "person_score": 6,  # 3 pronoun hits * 2
        "project_score": 0,
        "person_signals": ["pronoun nearby (3x)"],
        "project_signals": [],
    }
    result = classify_entity("Never", frequency=20, scores=scores)
    assert result["type"] == "uncertain"


# --- LICENSE/COPYING/etc. files are skipped ---


def test_scan_for_detection_skips_boilerplate_files(tmp_path):
    # Boilerplate that would otherwise classify "Software" / "Contributor" as entities
    (tmp_path / "LICENSE.md").write_text(
        "Software is provided 'as is'. Contributor grants rights. " * 50
    )
    (tmp_path / "real.md").write_text("Sam pushed the code. Sam laughed.")
    files = scan_for_detection(str(tmp_path), max_files=10)
    names = {Path(f).name for f in files}
    assert "LICENSE.md" not in names
    assert "real.md" in names


def test_skip_filenames_constant_includes_known_boilerplate():
    # Sanity check that the constant has the entries the README/commit message claims
    expected = {"license", "copying", "notice", "authors", "patents"}
    assert expected.issubset(SKIP_FILENAMES)


# --- New stopwords drop sentence-start participles ---


def test_extract_candidates_drops_participle_stopwords():
    text = " ".join(["Created the file. Updated the doc. Processed records."] * 5)
    candidates = extract_candidates(text)
    assert "Created" not in candidates
    assert "Updated" not in candidates
    assert "Processed" not in candidates
