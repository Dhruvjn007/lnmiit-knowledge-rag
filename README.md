# LNMIIT PYQ Search (RAG)

Search LNMIIT previous-year question papers with natural language.
Stack: **LangChain + FAISS + FastAPI**, Gemini (vision transcription + answers), local `bge-small` embeddings.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # add your GOOGLE_API_KEY
python ingest.py            # transcribe papers in question_papers/ and build the FAISS index
uvicorn app:app --reload    # http://127.0.0.1:8000/docs
```

## Endpoints
- `GET /search?q=distance vector routing&k=5&course_code=CC221`: matching past questions (no LLM call, fast)
- `POST /ask` `{"question": "What has been asked on Laurent series?"}`: grounded answer plus sources
- `GET /courses`: courses in the index

## Ingestion design (why it doesn't lose context)
Question papers are mostly scanned (no text layer), full of math, tables and figures, so plain text loaders plus fixed-size chunking break them.
1. **Vision transcription**: PDF pages are rendered to images. DOCX files are read in order, with text and embedded images interleaved. Both go to Gemini, which fills a Pydantic schema: paper metadata plus one entry per question.
2. **Question-level chunks**: each top-level question is one chunk, kept together with all its sub-parts. Math is written in LaTeX, and any figure gets a text description (graph edges and weights, block diagrams).
3. **Context header**: every chunk starts with `[course code, name | exam | semester]`, plus metadata used for filtering by course.
4. **Cache**: transcriptions are saved to `data/parsed/*.json`. Re-running only processes new papers (`--force` re-does all of them).

To add papers, drop them into `question_papers/` and run `python ingest.py` again.
