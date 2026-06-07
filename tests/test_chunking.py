"""Pure-logic tests (no API calls). / 순수 로직 테스트 (API 호출 없음)."""
from qa.indexer import chunk_text


def test_chunk_overlap():
    text = "abcdefghij" * 20  # 200 chars
    chunks = chunk_text(text, size=80, overlap=20)
    assert len(chunks) >= 3
    assert all(len(c) <= 80 for c in chunks)


def test_chunk_empty():
    assert chunk_text("", 100, 10) == []
