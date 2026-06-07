"""Retrieval: embed the question, find the nearest chunks in Chroma.
검색: 질문을 임베딩해 크로마에서 가까운 청크를 찾는다."""
import chromadb
from openai import OpenAI

from qa.models import RetrievedChunk
from qa.settings import settings

_openai = OpenAI(api_key=settings.openai_api_key)
_client = chromadb.PersistentClient(path=settings.chroma_dir)


def embed_query(query: str) -> list[float]:
    resp = _openai.embeddings.create(model=settings.embedding_model, input=query)
    return resp.data[0].embedding


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k
    coll = _client.get_collection(settings.collection_name)
    res = coll.query(query_embeddings=[embed_query(query)], n_results=top_k)
    out: list[RetrievedChunk] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append(
            RetrievedChunk(
                content=doc,
                source=str(meta.get("source", "?")),
                chunk_index=int(meta.get("chunk_index", 0)),
                distance=float(dist),
            )
        )
    return out
