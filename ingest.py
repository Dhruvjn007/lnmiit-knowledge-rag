"""
Ingest question papers -> FAISS index.

Why a vision LLM instead of a text loader:
  * Scanned PDFs have no text layer at all (PyPDFLoader returns empty strings).
  * Math (matrices, integrals), tables and figures (graphs, block diagrams) are
    lost or scrambled by OCR / naive text extraction.
  * Fixed-size chunking splits a question from its sub-parts and figure.

So each paper is sent (page images, or DOCX text+images in reading order) to
Gemini, which returns structured JSON: paper metadata + one entry per question
(with all sub-parts, LaTeX math, and a text description of any figure).
Each question becomes ONE chunk, prefixed with the course/exam header, so a
retrieved chunk is always self-contained.

Usage:  python ingest.py            (re-uses cached transcriptions in data/parsed)
        python ingest.py --force    (re-transcribe everything)
"""
import base64
import json
import sys

import pymupdf as fitz
from docx import Document as DocxDocument
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from common import INDEX_DIR, PAPERS_DIR, PARSED_DIR, get_embeddings, get_llm


# ---------- Output schema the LLM must fill ----------
class Question(BaseModel):
    number: str = Field(description="Question number as printed, e.g. '1', '5', 'Q.3'")
    text: str = Field(description="Full question text INCLUDING all sub-parts (a), (b), (i)... "
                                  "with their marks. Math in LaTeX ($...$).")
    marks: str = Field(description="Total marks for the question, e.g. '6' or '2+2+2'. Empty if not shown.")
    figure_description: str = Field(description="Detailed description of any figure/diagram/table the "
                                                "question refers to (nodes, edges, weights, blocks, values). "
                                                "Empty if none.")
    topics: list[str] = Field(description="2-5 syllabus topics this question tests, e.g. 'CSMA/CD', 'Laurent series'")


class Paper(BaseModel):
    course_code: str = Field(description="e.g. 'CC221', 'MT121', 'ECE 327'")
    course_name: str
    exam_type: str = Field(description="e.g. 'End-Term', 'Mid-Term I', 'Mid-Term II'")
    semester: str = Field(description="e.g. 'Even 2024-25', 'Autumn 2014-15'")
    max_marks: str
    duration: str
    questions: list[Question]


PROMPT = """You are transcribing an LNMIIT university exam question paper for a search engine.
Return the paper metadata and EVERY question.

Rules:
- Transcribe question text faithfully; do not solve or paraphrase.
- Keep each top-level question together with ALL its sub-parts (a), (b), (i), (ii)... in one entry.
- Write math in LaTeX ($...$), matrices with \\begin{bmatrix}.
- If a question refers to a figure, diagram, graph or table, describe it fully in figure_description
  (e.g. every node, edge and weight of a network graph; every block and signal in a block diagram)
  so the question is understandable without the image.
- Ignore the CO/PO weightage table, instructions and answer-sheet index tables.
- Some content may be images of equations embedded between text; merge them into the question text.
- If a symbol/equation is clearly missing from the source (blank gap), write [missing] — never invent it."""


# ---------- Turning files into multimodal message parts ----------
def _img_part(data: bytes, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def pdf_parts(path) -> list[dict]:
    # Render every page as an image; works for both scanned and digital PDFs.
    doc = fitz.open(path)
    return [_img_part(page.get_pixmap(dpi=200).tobytes("png"), "image/png") for page in doc]


def docx_parts(path) -> list[dict]:
    """Walk the DOCX body in reading order, interleaving text and embedded images
    (figures and equation images), so the LLM sees each image where it belongs."""
    doc = DocxDocument(path)
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    parts, buf = [], []

    def flush():
        if buf:
            parts.append({"type": "text", "text": "\n".join(buf)})
            buf.clear()

    for para in doc.element.body.iter(f"{W}p"):
        if next(para.iterancestors(f"{W}p"), None) is not None:
            continue  # text-box paragraph; already covered by its outer paragraph
        text = "".join(t.text or "" for t in para.iter() if t.tag in (f"{W}t", f"{M}t"))
        if text.strip():
            buf.append(text)
        # DrawingML images (a:blip r:embed) and legacy VML images (v:imagedata r:id)
        for el in para.iter():
            rid = el.get(f"{R}embed") or (el.get(f"{R}id") if el.tag.endswith("imagedata") else None)
            if not rid or rid not in doc.part.related_parts:
                continue
            img = doc.part.related_parts[rid]
            if img.content_type in ("image/png", "image/jpeg", "image/gif", "image/webp"):
                flush()
                parts.append(_img_part(img.blob, img.content_type))
    flush()
    return parts


def transcribe(path) -> dict:
    parts = pdf_parts(path) if path.suffix.lower() == ".pdf" else docx_parts(path)
    llm = get_llm().with_structured_output(Paper)
    paper: Paper = llm.invoke([HumanMessage(content=[{"type": "text", "text": PROMPT}, *parts])])
    return paper.model_dump()


# ---------- Paper JSON -> one LangChain Document per question ----------
def to_documents(paper: dict, source: str) -> list[Document]:
    header = f"{paper['course_code']} {paper['course_name']} | {paper['exam_type']} | {paper['semester']}"
    docs = []
    for q in paper["questions"]:
        content = f"[{header}]\nQuestion {q['number']}"
        content += f" ({q['marks']} marks)\n" if q["marks"] else "\n"
        content += q["text"]
        if q["figure_description"]:
            content += f"\nFigure: {q['figure_description']}"
        if q["topics"]:
            content += f"\nTopics: {', '.join(q['topics'])}"
        docs.append(Document(page_content=content, metadata={
            "source": source,
            "course_code": paper["course_code"],
            "course_name": paper["course_name"],
            "exam_type": paper["exam_type"],
            "semester": paper["semester"],
            "question_number": q["number"],
            "marks": q["marks"],
        }))
    return docs


def main(force: bool = False):
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    all_docs = []
    for path in sorted(PAPERS_DIR.iterdir()):
        if path.suffix.lower() not in (".pdf", ".docx"):
            continue
        cache = PARSED_DIR / f"{path.stem}.json"
        if cache.exists() and not force:
            paper = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[cached] {path.name}")
        else:
            print(f"[transcribing] {path.name} ...")
            paper = transcribe(path)
            cache.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
        docs = to_documents(paper, path.name)
        print(f"   -> {len(docs)} questions ({paper['course_code']} {paper['exam_type']})")
        all_docs.extend(docs)

    if not all_docs:
        sys.exit("No questions found. Put .pdf/.docx papers in question_papers/")
    FAISS.from_documents(all_docs, get_embeddings()).save_local(str(INDEX_DIR))
    print(f"Indexed {len(all_docs)} questions -> {INDEX_DIR}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
