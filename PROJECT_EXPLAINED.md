# LNMIIT PYQ Search: Full Project Explanation (Interview Prep)

---

## 1. The 30-second pitch

> "I built a RAG system that lets LNMIIT students search previous years' question papers in plain English, for example *'What have they asked about distance vector routing?'*, and get back the exact past questions, with course, exam and semester. It can also generate a grounded answer that cites the questions it used.
>
> The hard part was ingestion. Most of our papers are **scanned PDFs with no text layer**, full of matrices, integrals and diagrams. Standard text loaders return nothing or garbage. So I used a **vision LLM (Gemini) to transcribe each paper into structured JSON**, one entry per question, with math in LaTeX and figures described in text. Then I indexed **one chunk per question** in **FAISS** using local embeddings, and served it with **FastAPI**. The pipeline is orchestrated with **LangChain**."

---

## 2. The problem

- Students prepare for exams by digging through previous year question papers (PYQs) scattered across PDFs and Word files.
- They want to ask things like *"Which questions came on Laurent series?"* or *"CSMA/CD numericals"*. Keyword search (Ctrl+F) fails because:
  1. **Scanned PDFs have no text**, so there is nothing to Ctrl+F.
  2. The same concept is worded differently across years ("backoff", "capture effect", "collision").
- So we need **semantic search** over **correctly extracted** question content.

---

## 3. Architecture

```
                         INGESTION (offline, ingest.py)
 question_papers/                                                     data/
 ┌──────────────┐  render pages  ┌────────────┐  structured   ┌──────────────┐
 │ scanned PDFs │ ─────────────► │            │  JSON (Pydantic│ parsed/*.json│ (cache)
 │ (images)     │   200 dpi PNG  │  Gemini    │  schema)       └──────┬───────┘
 ├──────────────┤                │  vision    │ ──────────────►       │ one Document
 │ DOCX         │ ─────────────► │  LLM       │                       │ per question
 │ (text+images)│  interleaved   │            │                       ▼
 └──────────────┘  in order      └────────────┘               ┌──────────────┐
                                                bge-small     │ FAISS index  │
                                                embeddings ──►│ faiss_index/ │
                                                (local)       └──────┬───────┘
                                                                     │
                         SERVING (online, app.py, FastAPI)           │
   user ──► GET /search?q=...  ──► embed query ──► FAISS top-k ◄─────┘
                                                      │
                                                      ├──► return questions + metadata  (no LLM, fast)
   user ──► POST /ask         ──► same retrieval ─────┴──► prompt + context ──► Gemini ──► answer + sources
```

**Files**

| File | Role |
|---|---|
| `common.py` | Shared config: paths, embedding model, LLM factory, `.env` loading |
| `ingest.py` | Papers → Gemini transcription → per-question Documents → FAISS index |
| `app.py` | FastAPI service: `/search`, `/ask`, `/courses` |
| `data/parsed/*.json` | Cached transcriptions (one per paper) |
| `data/faiss_index/` | Saved FAISS index and docstore |

---

## 4. Ingestion: the most important part

### 4.1 What the raw data looks like (and why naive RAG fails)

I inspected the papers before writing any code:

| File | What I found | Naive loader result |
|---|---|---|
| `2024-25_CCE_221_ ET.pdf` (Computer Networks) | Scanned images, **0 characters of text**. Has a CO table, network graphs with edge weights (Fig. 1, Fig. 2), binary strings | Empty |
| `2024-25_Even_MT121_ET.pdf` (Linear Algebra & Complex Analysis) | Scanned, **0 text**. Has a 4×4 matrix, integrals, set notation | Empty |
| `CONTROL SYSTEM ENGINEERING 1st Term.docx` | Text plus **block diagrams and equations as embedded images** | Text with holes where equations and figures should be |
| `CONTROL SYSTEM ENGINEERING mid Term.docx` | Text plus figures, but **inline equations are actually missing from the file** (only whitespace, probably from an old PDF→Word conversion) | Holes that nothing can recover |

Even if OCR worked, **fixed-size chunking** (e.g. 500 characters) would:
- split Q5(a) from Q5(b), even though (b) depends on (a);
- separate a question from the figure it refers to;
- produce chunks with no idea which course or exam they came from.

### 4.2 My approach: "LLM as a structured parser"

**Step 1: Turn every file into multimodal input**
- **PDF**: each page is rendered to a PNG at 200 DPI with PyMuPDF. This works for both scanned and digital PDFs, so there is one code path for both.
- **DOCX**: I walk the Word XML body **in reading order** and build an interleaved list: `text, image, text, image...`. That way the LLM sees each equation or figure image *exactly where it sits* in the question. Details:
  - Text comes from both `w:t` (normal text) and `m:t` (Office Math) elements.
  - Images come from both DrawingML (`a:blip r:embed`) and legacy VML (`v:imagedata r:id`).
  - Paragraphs nested inside text boxes are skipped, because their outer paragraph already includes them. Otherwise text would be duplicated.

**Step 2: One Gemini call per paper with structured output**
- LangChain's `llm.with_structured_output(Paper)` forces the response to match a **Pydantic schema**:
  ```
  Paper:    course_code, course_name, exam_type, semester, max_marks, duration, questions[]
  Question: number, text (all sub-parts, LaTeX math), marks, figure_description, topics[]
  ```
- Prompt rules: transcribe faithfully (don't solve), **keep sub-parts together**, use LaTeX for math, **describe figures fully** (every node/edge/weight, every block), ignore CO/PO tables and instructions, and write `[missing]` for blank gaps instead of inventing symbols (this guards against hallucination).
- Why the whole paper in one call rather than page by page: a question can **span a page break** (e.g. MT121 Q5 continues onto page 2). The LLM sees the full paper, so it can stitch the question back together.

**Step 3: One chunk per question, with a context header**
```
[CC221 Computer Communication Networks | End-Term | Even 2024-25]
Question 5 (2+4 marks)
(a) Consider the network with link costs shown in Fig.1. Use Distance Vector ...
(b) Let's assume the minimum cost to destination node 4 ... link between 3 and 4 goes down ...
Figure: Undirected graph with nodes 1-6; edges 1-2 (3), 1-3 (2), 3-5 (3), ...
Topics: distance vector routing, count-to-infinity, routing tables
```
- **The question is the natural semantic unit** of a question paper, so chunk boundaries follow meaning, not character counts.
- The **header** goes into the embedded text, so a query like "CN end-term routing question" matches on course and exam too. It also makes each search result understandable on its own.
- **Figure descriptions** make diagram-based questions searchable ("graph with 6 nodes", "block diagram feedback").
- **Topics** generated by the LLM act as **query expansion at index time**: a question that never says "count-to-infinity" can still be found by that term.
- The same fields are also stored as **metadata** (`course_code`, `exam_type`, `semester`, `question_number`, `marks`, `source`) for filtering and citations.

**Step 4: Caching**
- Each transcription is saved as `data/parsed/<paper>.json`. Re-running `ingest.py` only calls the LLM for **new** papers (`--force` redoes all of them).
- Benefits: it saves API cost and time, the index can be rebuilt (e.g. with a different embedding model) without re-transcribing, and the JSON can be **inspected and hand-corrected**.

### 4.3 Why a vision LLM instead of Tesseract OCR + text loaders?

| | Tesseract / OCR | Vision LLM (chosen) |
|---|---|---|
| Math (matrices, integrals, fractions) | Garbled | LaTeX |
| Figures (graphs, block diagrams) | Ignored | Described in text |
| Question / sub-part structure | Lost; needs regex heuristics | Native structured JSON |
| Layout (two columns: text + figure side by side, as in CC221 Q5) | Lines from both columns get mixed together | Understood |
| Cost | Free | A few API calls per paper (Gemini free tier is enough) |
| Determinism | Deterministic | Small risk of hallucination, mitigated by temperature 0, a "transcribe, don't solve" prompt, `[missing]` markers and the inspectable JSON cache |

For a one-time ingestion of a few hundred papers, accuracy matters far more than per-page cost.

---

## 5. Indexing and retrieval

### Embeddings: `BAAI/bge-small-en-v1.5` via FastEmbed (local)
- **384-dimensional**, ~130 MB, runs on CPU through ONNX Runtime. It **doesn't need PyTorch** and makes no API calls.
- Strong for its size on retrieval benchmarks (MTEB) and fast enough for real-time query embedding.
- Its 512-token limit is fine, because a single exam question is well under that.
- Trade-off: a larger or API embedding model might retrieve slightly better, but local means free, private and no network latency per query.

### Vector store: FAISS
- LangChain's `FAISS.from_documents` builds an **`IndexFlatL2`**: exact, brute-force nearest neighbour search.
- At this scale (hundreds to a few thousand questions) exact search takes **under a millisecond**, so approximate indexes (IVF, HNSW) aren't worth their complexity or recall loss. They become worth it at hundreds of thousands to millions of vectors.
- bge embeddings are normalized, so **L2 distance ranks results the same way cosine similarity does**. The API returns the L2 distance as `score` (lower is better).
- Saved to disk with `save_local` and loaded once at API startup. `allow_dangerous_deserialization=True` is needed because the docstore is pickled. That is safe here only because **we built the index ourselves**; never load an untrusted one.
- **Why FAISS rather than a vector DB** (Pinecone, Chroma, pgvector): the corpus is small and mostly static, so an in-process library with no server keeps it simple. For production with frequent updates or multiple instances, I'd move to a managed or DB-backed store.

### Metadata filtering
- `/search?course_code=CC221` passes `filter={"course_code": "CC221"}`.
- LangChain's FAISS wrapper **post-filters**: it fetches `fetch_k` candidates (I use `max(20, 4k)`) and then filters them. So I raise `fetch_k`; otherwise a filter could return fewer than k results.

---

## 6. Serving: FastAPI (`app.py`)

| Endpoint | What it does | LLM call? |
|---|---|---|
| `GET /search?q=&k=5&course_code=` | Embed query → FAISS top-k → return question text, metadata and score | No (fast, free) |
| `POST /ask` `{question, k, course_code}` | Same retrieval → stuff the k questions into a prompt → Gemini answer with citations, plus sources | Yes |
| `GET /courses` | Distinct course codes and names in the index (for a UI dropdown or filter) | No |

- **Why two endpoints:** most of the time a student just wants the matching past questions, and that should be instant and cost nothing. `/ask` is for summaries ("what topics repeat in MT121?").
- **The RAG chain** is LangChain Expression Language (LCEL): `ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`.
- **Grounding prompt:** "Answer using ONLY the past questions below. Cite (course code, exam, semester, Q number). If none are relevant, say so." This reduces hallucination and makes answers verifiable.
- **Pydantic models** (`Hit`, `AskRequest`, `AskResponse`) give request validation and **automatic Swagger docs** at `/docs`.
- The index and embedding model are loaded **once at startup**, not per request.
- The **"stuff" strategy**: all k chunks go into one prompt. That works because each chunk is one question (small), so 5 chunks fit easily in context.

---

## 7. Tech stack: what and why

| Tool | Used for | Why |
|---|---|---|
| **LangChain** | Gemini wrapper, structured output, Documents, FAISS wrapper, prompt templates, LCEL chain | One consistent interface; swapping LLM, embeddings or vector store is a one-line change |
| **Gemini** (`langchain-google-genai`) | Vision transcription and answer generation | Multimodal, good at math/LaTeX, **free tier**. Model name is configurable via the `GEMINI_MODEL` env var |
| **FAISS** (`faiss-cpu`) | Vector index | Fast, in-process, no server, persists to disk |
| **FastEmbed + bge-small** | Embeddings | Local, free, light (no torch) |
| **FastAPI + Uvicorn** | REST API | Async-ready, Pydantic validation, auto docs |
| **PyMuPDF** | Render PDF pages to images | Fast, no external dependencies (no poppler) |
| **python-docx** | Read DOCX XML and embedded images | Direct access to the XML tree for reading-order extraction |
| **Pydantic** | Schema for LLM output and API models | Validated, typed structured data |

---

## 8. Data flow example (end to end)

1. A student calls `GET /search?q=how does receiver verify checksum`.
2. FastEmbed turns the query into a 384-dimensional vector.
3. FAISS finds the 5 nearest question vectors.
4. Top hit: `[CC221 ... | End-Term | Even 2024-25] Question 2 (2+3 marks) The three 16-bit words at the sender are ... calculating the checksum? How will the receiver verify ...` with `source: 2024-25_CCE_221_ ET.pdf`.
5. With `/ask`, those 5 questions plus the student's question go to Gemini, which replies with something like *"Checksum questions appeared in CC221 End-Term 2024-25 (Q2), asking you to compute a 16-bit one's-complement checksum and explain receiver verification..."*.

---

## 9. Design decisions and trade-offs (likely interview questions)

**Q: Why not just use `PyPDFLoader` + `RecursiveCharacterTextSplitter`?**
I checked the data first. The PDFs have zero extractable text, and even for text documents, character-based splitting breaks questions apart from their sub-parts and figures. The data's natural unit is the question, so I chunk by question.

**Q: Isn't one LLM call per paper expensive?**
It's a one-time cost per paper, cached as JSON. A paper is 2–4 pages, so a few hundred papers cost little (or fit in the free tier). Queries through `/search` never call the LLM.

**Q: How do you handle LLM hallucination during transcription?**
Temperature 0, an explicit "transcribe, don't solve or paraphrase" instruction, a strict Pydantic schema, `[missing]` for unreadable parts, and a human-readable JSON cache that can be spot-checked and corrected before indexing.

**Q: What if a question spans two pages?**
All pages of a paper go in one call, so the LLM sees the continuation and merges it into one question.

**Q: Why put metadata in the chunk text as well as in metadata?**
Metadata alone is used for filtering but isn't embedded. Putting the header in the text lets semantic search match "networks end-term" and makes each retrieved chunk self-explanatory to the answering LLM.

**Q: Why local embeddings but a cloud LLM?**
Embeddings are needed on *every* query, so local means zero latency and zero cost. The LLM is only needed at ingestion and for the optional `/ask`, where its quality matters.

**Q: How would you scale this to all LNMIIT papers (thousands)?**
- Ingestion: run transcriptions in parallel (async/batch) with rate-limit handling and retries.
- Index: FAISS flat search is still fine at ~10k–50k questions. Beyond that, use HNSW, or a vector DB with native metadata filtering.
- Incremental updates: `FAISS.add_documents` for new papers instead of rebuilding.
- Serving: multiple Uvicorn workers; the index is read-only, so it's easy to replicate.

**Q: How would you improve retrieval quality?**
- **Hybrid search**: BM25 plus vectors (LangChain `EnsembleRetriever`) for exact terms like course codes, "CSMA/CD" or "MTU".
- **Re-ranking**: a cross-encoder (e.g. `bge-reranker`) over the top 20.
- **Better filters**: year, exam type, marks.
- A bigger embedding model if evaluation shows it helps.

**Q: How would you evaluate it?**
- **Transcription**: manually compare the JSON for a sample of papers against the originals (are questions complete, is the math correct, are sub-parts merged correctly?).
- **Retrieval**: build a small labelled set (query → expected question IDs) and measure **Recall@k / MRR**. Use it to compare chunking strategies, embedding models, and hybrid vs dense search.
- **Answers**: check faithfulness (does each claim cite a retrieved question?), for example with an LLM-as-judge or RAGAS.

**Q: What are the security considerations?**
- The API key lives in `.env`, which is gitignored.
- `allow_dangerous_deserialization` is acceptable only because the index is self-built.
- In production, the API would need auth and rate limits on `/ask` to protect the LLM quota.

---

## 10. Limitations (be honest about these in interviews)

- **Source-data loss:** the Control Systems mid-term DOCX has equations that are missing from the file itself. The system marks them `[missing]`, and the real fix is a better source scan.
- Transcription quality depends on scan quality and the LLM. Handwritten or very blurry scans may need review.
- Figures are searchable only through their *text description*; the images themselves aren't stored or returned.
- FAISS post-filtering can return fewer than k results under a narrow filter if the matches aren't among the `fetch_k` candidates.
- No UI yet. Swagger `/docs` serves as the interactive demo.

## 11. Future work
- A simple frontend (Streamlit/React) with course and year filters, rendering LaTeX with KaTeX.
- Return the page image or crop alongside each question.
- Hybrid BM25 + vector retrieval, and re-ranking.
- "Topic frequency" analytics: which topics repeat most across years (uses the `topics` field already extracted).
- Auto-ingest when a new paper is uploaded.

---

## 12. How to run / demo

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # set GOOGLE_API_KEY (free at aistudio.google.com/apikey)
python ingest.py                # transcribe + index
uvicorn app:app --reload        # open http://127.0.0.1:8000/docs
```

**Demo script for the interview:**
1. Open a scanned PDF and show it's an image (select-all copies no text): *"This is why naive RAG fails."*
2. Show a JSON file in `data/parsed/`, with the matrix in LaTeX and the network graph described as text.
3. In `/docs`, try `GET /search?q=count to infinity problem`. It finds the distance-vector question even though it never says "count to infinity" (through the figure description and topics).
4. Try `POST /ask` with *"What has been asked about Laurent series and singularities?"* to get a cited answer.
5. Use `course_code=MT121` to show metadata filtering.

---

## 13. One-line summary of each component (cheat sheet)

- **Ingestion:** vision LLM → Pydantic-structured JSON → one chunk per question, with a course/exam header, LaTeX math, figure descriptions and topics.
- **Embeddings:** bge-small, local, 384-dimensional, normalized.
- **Index:** FAISS IndexFlatL2, exact search, persisted to disk.
- **Retrieval:** top-k similarity search with an optional metadata filter (post-filter with `fetch_k`).
- **Generation:** LCEL chain, a grounded prompt that requires citations, and Gemini.
- **API:** FastAPI with `/search` (no LLM), `/ask` (RAG) and `/courses`.
