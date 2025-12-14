# =========================================================
# IMPORTS (MUST BE FIRST — DO NOT MOVE)
# =========================================================

import os
import sys
import json
import re
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# Ensure project root is on sys.path so `src` imports work
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.main import answer_question, ingest_and_index_documents
from src.llm.client import LLMClient

# =========================================================
# HELPERS
# =========================================================

def extract_json_array(raw: str) -> list:
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        raise ValueError("Model did not return a JSON array.")
    return json.loads(match.group(0))


def confidence_badge(label: str) -> str:
    return {"high": "✅ High", "medium": "🟠 Medium"}.get(label, "⚪ Low")


def kb_raw_path(filename: str) -> str:
    return os.path.join("data", "kb_raw", filename)


def read_pdf_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _safe_read(path: str, mode="r"):
    with open(path, mode) as f:
        return f.read()


# =========================================================
# PAGE CONFIG + SESSION STATE
# =========================================================

st.set_page_config(page_title="PolicyNavigator AI", page_icon="📘", layout="wide")

for k, v in {
    "kb_files": [],
    "kb_indexed": False,
    "quiz_items": None,
    "last_answer": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# STYLING (UNCHANGED)
# =========================================================

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; }
.pn-card {
  background: white;
  border-radius: 16px;
  padding: 16px;
  border: 1px solid #eee;
  box-shadow: 0 10px 24px rgba(0,0,0,0.05);
}
.pn-answer { border: 1px solid #eee; border-radius: 16px; padding: 16px; }
.pdf-viewer { border: 3px solid #ff7a00; border-radius: 12px; padding: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📚 Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload policy documents", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

if st.sidebar.button("Index Documents", type="primary"):
    if not uploaded_files:
        st.sidebar.warning("Upload files first.")
    else:
        os.makedirs("data/kb_raw", exist_ok=True)
        paths = []
        for f in uploaded_files:
            p = os.path.join("data/kb_raw", f.name)
            with open(p, "wb") as out:
                out.write(f.read())
            paths.append(p)

        n = ingest_and_index_documents(paths)
        st.session_state.kb_files = [f.name for f in uploaded_files]
        st.session_state.kb_indexed = True
        st.sidebar.success(f"Indexed {n} chunks")

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Project Links")
st.sidebar.markdown(
    """
- **GitHub Repo:**  
  https://github.com/akshayagavhane9/policynavigator-ai

- **Documentation:**  
  `docs/project_documentation.pdf`

- **Architecture Diagram:**  
  `docs/architecture_diagram.png`

- **🎥 10-Minute Walkthrough:**  
  https://drive.google.com/file/d/19p0mJd6X-LtVWKXl-YT6B0tjewDmqusY/view
"""
)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="pn-card">
<h1>📘 PolicyNavigator AI</h1>
<p>RAG-powered assistant for understanding complex policies (not limited to universities).</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# TABS
# =========================================================

tab_qa, tab_whatif, tab_quiz, tab_eval, tab_docs = st.tabs(
    ["💬 Q&A", "🤔 What-If", "📝 Quiz", "📈 Evaluation", "📘 About"]
)

# =========================================================
# TAB 1 — Q&A
# =========================================================

with tab_qa:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)

    style = st.radio(
        "Answer style",
        ["Strict policy quote", "Explain my rights (student-friendly)"],
        horizontal=True,
    )

    question = st.text_input("Ask a question")

    if st.button("Generate Answer", type="primary"):
        res = answer_question(question, answer_style=style)
        if (
            style.startswith("Explain")
            and res["answer"] == "Not covered in the provided policy excerpts."
        ):
            res["answer"] = (
                "The policy excerpts do not directly answer this. "
                "Try adding context (exam vs assignment, collaboration rules), "
                "or upload the specific policy section."
            )
        st.session_state.last_answer = res

    res = st.session_state.last_answer
    if res:
        st.markdown("### 🧠 Answer")
        st.write(res["answer"])

        st.markdown("### 📎 Evidence")
        for c in res["citations"]:
            with st.expander(
                f"{c['source']} | page {c.get('page')} | sim={c['similarity']:.2f}"
            ):
                st.markdown(c["text"][:1200])

                if c["source"].lower().endswith(".pdf"):
                    pdf_path = kb_raw_path(c["source"])
                    if os.path.exists(pdf_path):
                        b64 = read_pdf_as_base64(pdf_path)
                        components.html(
                            f"""
                            <div class="pdf-viewer">
                            <object data="data:application/pdf;base64,{b64}#page={c.get('page',1)}"
                                    type="application/pdf"
                                    width="100%" height="600">
                            </object>
                            </div>
                            """,
                            height=620,
                        )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2 — WHAT IF
# =========================================================

with tab_whatif:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
    scenario = st.text_area("Describe a scenario")

    if st.button("Analyze", type="primary"):
        llm = LLMClient()
        out = llm.chat(
            "You are a policy advisor.",
            f"Analyze this scenario and explain relevant policies:\n{scenario}",
        )
        st.write(out)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 — QUIZ
# =========================================================

with tab_quiz:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)

    if st.button("Generate Quiz", type="primary"):
        llm = LLMClient()
        raw = llm.chat(
            "Generate quiz",
            "Create 5 multiple-choice policy questions in JSON only.",
        )
        st.session_state.quiz_items = extract_json_array(raw)

    for i, q in enumerate(st.session_state.quiz_items or [], 1):
        st.markdown(f"**Q{i}. {q['question']}**")
        st.radio("Choose", q["options"], key=f"q{i}")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 4 — EVALUATION
# =========================================================

with tab_eval:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)

    csv_path = "results/ab_eval_runs.csv"
    json_path = "results/ab_eval_summary.json"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)

    if os.path.exists(json_path):
        st.json(json.loads(_safe_read(json_path)))

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 5 — ABOUT
# =========================================================

with tab_docs:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
    st.markdown(
        """
**PolicyNavigator AI**  
A modular RAG system for *any* policy domain.

**Key components**
- Document ingestion (PDF/DOCX/TXT)
- Semantic chunking
- Vector search + MMR
- Hallucination guardrails
- A/B evaluation framework
- Web + Streamlit UI

Built by **Akshaya & Ritwik**.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
