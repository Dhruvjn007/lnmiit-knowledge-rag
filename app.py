"""
FastAPI service over the FAISS index of past LNMIIT question papers.

  GET  /search?q=...   -> matching past questions (pure vector search, no LLM, fast)
  POST /ask            -> LLM answer grounded in the retrieved questions, with sources

Run:  uvicorn app:app --reload     then open http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from common import INDEX_DIR, get_embeddings, get_llm

if not INDEX_DIR.exists():
    raise RuntimeError("Index not found. Run `python ingest.py` first.")

# Loaded once at startup. allow_dangerous_deserialization is safe here: we built this index ourselves.
store = FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You help LNMIIT students explore previous years' exam questions. "
     "Answer using ONLY the past questions below. Cite each one you use as "
     "(course code, exam, semester, Q number). If none are relevant, say so.\n\n{context}"),
    ("human", "{question}"),
])
chain = prompt | get_llm(temperature=0.2) | StrOutputParser()

app = FastAPI(title="LNMIIT PYQ Search")


class Hit(BaseModel):
    score: float  # L2 distance: lower = more similar
    content: str
    metadata: dict


class AskRequest(BaseModel):
    question: str
    k: int = 5
    course_code: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Hit]


def retrieve(query: str, k: int, course_code: str | None) -> list[Hit]:
    flt = {"course_code": course_code} if course_code else None
    results = store.similarity_search_with_score(query, k=k, filter=flt, fetch_k=max(20, k * 4))
    return [Hit(score=float(s), content=d.page_content, metadata=d.metadata) for d, s in results]


@app.get("/search", response_model=list[Hit])
def search(q: str, k: int = 5, course_code: str | None = None):
    return retrieve(q, k, course_code)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    hits = retrieve(req.question, req.k, req.course_code)
    if not hits:
        raise HTTPException(404, "No matching questions found.")
    context = "\n\n---\n\n".join(h.content for h in hits)
    return AskResponse(answer=chain.invoke({"context": context, "question": req.question}), sources=hits)


@app.get("/courses")
def courses():
    """Distinct courses in the index (useful for the course_code filter)."""
    seen = {(d.metadata["course_code"], d.metadata["course_name"]) for d in store.docstore._dict.values()}
    return [{"course_code": c, "course_name": n} for c, n in sorted(seen)]
