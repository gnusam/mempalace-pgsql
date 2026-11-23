"""Tests for mempalace.general_extractor (emotion markers)."""

from mempalace.general_extractor import ALL_MARKERS, _score_markers, extract_memories


# --- Upstream 2d4e1085 (PR #2199): markdown emphasis is not an emotion ---


def test_markdown_emphasis_is_not_an_emotion_marker():
    assert r"\*[^*]+\*" not in ALL_MARKERS["emotional"]


def test_markdown_emphasis_does_not_score_as_emotional():
    text = (
        "**Key reasons:** transactional guarantees and mature tooling. "
        "The rollout plan is documented under *Migration notes* and **Schema**."
    )
    score, _ = _score_markers(text.lower(), ALL_MARKERS["emotional"])
    assert score == 0.0


def test_markdown_bold_does_not_override_a_real_decision():
    text = (
        "We decided to use PostgreSQL for the new service. "
        "**Key reasons:** transactional guarantees and mature tooling. "
        "The rollout details are documented under **Migration notes**."
    )
    memories = extract_memories(text, min_confidence=0.1)
    assert len(memories) == 1
    assert memories[0]["memory_type"] == "decision"


def test_real_emotion_words_inside_emphasis_still_count():
    score, _ = _score_markers("i feel *sad* today and i miss home", ALL_MARKERS["emotional"])
    assert score > 0
