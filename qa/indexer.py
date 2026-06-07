"""Indexing: documents -> chunks -> embeddings -> Chroma.
인덱싱: 문서 -> 청크 -> 임베딩 -> 크로마 저장."""
import hashlib
from pathlib import Path

import chromadb
from openai import OpenAI

from qa.settings import settings

_openai = None
_client = None


def _oa():
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=settings.openai_api_key)
    return _openai


def _chroma():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _client


def load_text(path: Path) -> str:
    if path.suffix.lower() in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    raise ValueError(f"unsupported file type: {path}")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size chunking with overlap. / 고정 크기 + 중첩 청킹."""
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    resp = _oa().embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def index_folder(docs_dir: str = "./docs") -> int:
    """Read every file in docs_dir, chunk + embed, store in Chroma. Returns chunk count."""
    coll = _chroma().get_or_create_collection(settings.collection_name)
    paths = [p for p in Path(docs_dir).glob("**/*") if p.suffix.lower() in (".md", ".txt", ".pdf")]
    total = 0
    for path in paths:
        chunks = chunk_text(load_text(path), settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            continue
        embeddings = embed(chunks)
        ids = [hashlib.md5(f"{path.name}-{i}".encode()).hexdigest() for i in range(len(chunks))]
        coll.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": path.name, "chunk_index": i} for i in range(len(chunks))],
        )
        total += len(chunks)
        print(f"  indexed {path.name}: {len(chunks)} chunks")
    return total


if __name__ == "__main__":
    n = index_folder()
    print(f"done: {n} chunks indexed into '{settings.collection_name}'")
