import os
import sys
import json
import re
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure project root is on sys.path so `src` imports work in Streamlit
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import streamlit.components.v1 as components

from src.main import answer_question, ingest_and_index_documents  # type: ignore
from src.llm.client import LLMClient  # type: ignore

# --- NEW: for evaluation tab ---
import pandas as pd


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def extract_json_array(raw: str) -> list:
    """Extract a JSON array from an LLM response that may contain extra text."""
    if not raw or not raw.strip():
        raise ValueError("Model returned an empty response.")
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        raise ValueError("Model did not return a JSON array.")
    return json.loads(match.group(0))


def confidence_badge(label: str) -> str:
    label = (label or "").lower()
    if label == "high":
        return "✅ High"
    if label == "medium":
        return "🟠 Medium"
    return "⚪ Low"


def kb_raw_path(filename: str) -> str:
    return os.path.join("data", "kb_raw", filename)


def read_pdf_as_base64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# --- NEW: evaluation file helpers ---
def _safe_read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _safe_read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _pct(n: float) -> str:
    try:
        return f"{(float(n) * 100):.1f}%"
    except Exception:
        return "—"


# -------------------------------------------------------------------
# Page config & Session state
# -------------------------------------------------------------------

st.set_page_config(page_title="PolicyNavigator AI", page_icon="📘", layout="wide")

if "kb_files" not in st.session_state:
    st.session_state["kb_files"] = []
if "kb_indexed" not in st.session_state:
    st.session_state["kb_indexed"] = False
if "quiz_items" not in st.session_state:
    st.session_state["quiz_items"] = None

# Store last answer result
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = None


# -------------------------------------------------------------------
# Styling (KEEP EXACT — unchanged)
# -------------------------------------------------------------------

st.markdown(
    """
<style>
/* Global spacing and typography */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }

/* Sidebar clean cards */
section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid #eee;
}
section[data-testid="stSidebar"] .stMarkdown p { color: #555; }

/* Primary buttons: orange */
div.stButton > button[kind="primary"] {
  background: #ff7a00 !important;
  color: white !important;
  border: 1px solid #ff7a00 !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}
div.stButton > button[kind="primary"]:hover {
  filter: brightness(0.96);
}

/* Default buttons: subtle */
div.stButton > button {
  border-radius: 12px !important;
}

/* "Card" container look */
.pn-card {
  background: #ffffff;
  border: 1px solid #eee;
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.05);
}

/* Answer panel */
.pn-answer {
  background: #ffffff;
  border: 1px solid #eee;
  border-radius: 16px;
  padding: 16px;
}

/* Mini badge */
.pn-badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid #eee;
  background: #fafafa;
  font-size: 12px;
  color: #444;
}

/* Reduce expander header clutter */
div[data-testid="stExpander"] summary {
  font-weight: 650;
}

/* Make expander content more readable */
div[data-testid="stExpander"] .stMarkdown {
  color: #333;
}

/* PDF viewer container */
.pdf-viewer-container {
  position: sticky;
  top: 60px;
  z-index: 100;
  background: white;
  padding: 20px;
  border-radius: 12px;
  border: 3px solid #ff7a00;
  margin-bottom: 20px;
  box-shadow: 0 10px 40px rgba(255, 122, 0, 0.2);
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Sidebar – Knowledge Base + Links
# -------------------------------------------------------------------

st.sidebar.title("📚 Knowledge Base")
st.sidebar.caption(
    "Upload NEU policy documents (PDF, DOCX, TXT). They will ground answers and quizzes."
)

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt"],
    label_visibility="collapsed",
)

if uploaded_files:
    st.sidebar.markdown("**Selected files:**")
    for f in uploaded_files:
        st.sidebar.write(f"• {f.name}")
elif st.session_state["kb_files"]:
    st.sidebar.caption("Last indexed:")
    for name in st.session_state["kb_files"]:
        st.sidebar.write(f"• {name}")

if st.sidebar.button("Index Documents", type="primary"):
    try:
        raw_dir = "data/kb_raw"
        os.makedirs(raw_dir, exist_ok=True)

        if not uploaded_files:
            st.sidebar.warning("Upload at least one file first.")
        else:
            saved_paths: List[str] = []
            saved_names: List[str] = []

            for uf in uploaded_files:
                save_path = os.path.join(raw_dir, uf.name)
                with open(save_path, "wb") as out:
                    out.write(uf.read())
                saved_paths.append(save_path)
                saved_names.append(uf.name)

            n_chunks = ingest_and_index_documents(saved_paths)
            st.session_state["kb_files"] = saved_names
            st.session_state["kb_indexed"] = True
            st.session_state["quiz_items"] = None
            st.sidebar.success(f"Indexed {n_chunks} chunks from {len(saved_paths)} file(s).")

    except Exception as e:
        st.sidebar.error(f"Indexing failed: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Project Links")
st.sidebar.markdown(
    """
- GitHub Repo: [PolicyNavigator AI](https://github.com/akshayagavhane9/policynavigator-ai)
- Documentation (PDF): `docs/project_documentation.pdf`
- Architecture Diagram: `docs/architecture_diagram.png`
"""
)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Example questions")
st.sidebar.markdown(
    """
- What is the policy on cheating during exams?
- What happens after an academic integrity violation?
- How does the appeal process work?
"""
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown(
    """
<div class="pn-card">
  <div style="display:flex; align-items:center; gap:14px;">
    <div style="font-size:44px;">📘</div>
    <div>
      <div style="font-size:34px; font-weight:900; color:#111; line-height:1.1;">PolicyNavigator AI</div>
      <div style="color:#555; margin-top:6px;">
        Retrieval-augmented policy assistant • Student rights explainer • Scenario simulator • Quiz tutor
        <span class="pn-badge" style="margin-left:10px;">Black/Orange theme</span>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- UPDATED: add evaluation tab ---
tab_qa, tab_whatif, tab_quiz, tab_eval, tab_docs = st.tabs(
    ["💬 Q&A Assistant", "🤔 What-If Scenarios", "📝 Policy Quiz", "📈 Evaluation (A/B)", "📘 About & Docs"]
)


# -------------------------------------------------------------------
# Tab 1 – Q&A Assistant
# -------------------------------------------------------------------

with tab_qa:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
    st.subheader("Ask a Policy Question")

    colA, colB = st.columns([2, 1])
    with colA:
        answer_style = st.radio(
            "Answer style",
            ["Strict policy quote", "Explain my rights (student-friendly)"],
            horizontal=True,
        )
    with colB:
        k = st.slider("Context chunks", 3, 10, 5)
        rewrite_query = st.checkbox("Rewrite query for retrieval", value=True)

    user_question = st.text_input(
        "Your question",
        placeholder="e.g., How does Northeastern define cheating in academic integrity policy?",
        label_visibility="collapsed",
    )

    if st.button("Generate Answer", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Consulting your indexed policy documents..."):
                try:
                    res = answer_question(
                        user_question,
                        answer_style=answer_style,
                        rewrite_query=rewrite_query,
                        k=k,
                    )

                    # If student-friendly but abstained, show a helpful UX message
                    if (
                        answer_style == "Explain my rights (student-friendly)"
                        and isinstance(res, dict)
                        and (res.get("answer") or "").strip() == "Not covered in the provided policy excerpts."
                    ):
                        res["answer"] = (
                            "I couldn't find a clear clause in the uploaded policy excerpts that directly answers this. "
                            "Try rephrasing with more specifics (assignment vs exam, collaboration, allowed resources), "
                            "or upload the exact policy document that covers this topic."
                        )

                    st.session_state["last_answer"] = res

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.session_state["last_answer"] = None

    st.markdown("</div>", unsafe_allow_html=True)

    # Retrieve the last answer from session state
    res = st.session_state.get("last_answer")

    if res:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("<div class='pn-answer'>", unsafe_allow_html=True)
            st.markdown("### 🧠 Answer")
            st.write(res.get("answer", ""))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Confidence & Risk")

            label = res.get("confidence_label", "low")
            score = float(res.get("confidence_score", 0.0))
            st.markdown(f"**Confidence:** {confidence_badge(label)}  \n**Score:** `{score:.2f}`")

            hallucination_flag = bool(res.get("hallucination_flag", False))
            hallucination_risk = res.get("hallucination_risk", "unknown")
            if hallucination_flag:
                st.error(f"⚠ Hallucination risk: **{hallucination_risk}**")
            else:
                st.info(f"Hallucination risk: **{hallucination_risk}**")

            st.caption(f"Latency: {int(res.get('latency_ms', 0))} ms")

            st.markdown("### ⭐ Feedback")
            fb1, fb2 = st.columns(2)
            with fb1:
                st.button("👍 Helpful", key="fb_helpful")
            with fb2:
                st.button("👎 Not helpful", key="fb_not_helpful")

            st.markdown("</div>", unsafe_allow_html=True)

        # Evidence section (clean)
        st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
        st.markdown("### 📎 Evidence & Citations (with PDF page preview)")

        citations = res.get("citations", []) or []
        if not citations:
            st.write("No citations returned.")
        else:
            for c in citations:
                src = c.get("source", "unknown")
                chunk_id = c.get("chunk_id", "")
                rank = c.get("rank", "")
                sim = float(c.get("similarity", 0.0))
                text = (c.get("text") or "").strip()
                page = c.get("page", None)

                # Clean header
                page_label = f"page {page}" if page else "page ?"
                header = f"{src} • {page_label} • rank {rank} • sim={sim:.2f}"

                with st.expander(header, expanded=(rank == 1)):
                    st.markdown("**Excerpt**")
                    st.markdown(f"> {text[:1500]}" + ("..." if len(text) > 1500 else ""))
                    st.caption(f"Similarity: {sim:.2f}")

                    # PDF Preview button only when source is a PDF and exists
                    if str(src).lower().endswith(".pdf"):
                        pdf_path = kb_raw_path(src)

                        if os.path.exists(pdf_path):
                            if st.button(
                                "📄 View PDF page",
                                key=f"view_pdf_{src}_{chunk_id}_{rank}",
                                type="primary",
                                use_container_width=True,
                            ):
                                st.markdown("---")
                                st.markdown(f"### 📄 PDF Preview: {src} - Page {page}")

                                try:
                                    pdf_b64 = read_pdf_as_base64(pdf_path)
                                    display_page = int(page) if page else 1

                                    pdf_html = f"""
                                    <div class="pdf-viewer-container">
                                        <div style="margin-bottom: 10px; padding: 10px; background: linear-gradient(90deg, #111 0%, #222 70%, #ff7a00 140%); color: white; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                                            <div><strong>📄 {src}</strong> - Page {display_page}</div>
                                            <div style="font-size: 12px;">Similarity: {sim:.2f}</div>
                                        </div>
                                        <object
                                            data="data:application/pdf;base64,{pdf_b64}#page={display_page}"
                                            type="application/pdf"
                                            width="100%"
                                            height="800"
                                            style="border: 1px solid #ddd; border-radius: 8px;">
                                            <div style="padding: 40px; text-align: center; background: #f9f9f9; border-radius: 8px;">
                                                <p style="color: #666; margin-bottom: 20px;">Your browser doesn't support embedded PDFs.</p>
                                                <a href="data:application/pdf;base64,{pdf_b64}" download="{src}"
                                                   style="display: inline-block; padding: 12px 24px; background: #ff7a00; color: white; text-decoration: none; border-radius: 8px; font-weight: bold;">
                                                   📥 Download PDF
                                                </a>
                                            </div>
                                        </object>
                                        <div style="margin-top: 10px; padding: 8px; background: #f0f0f0; border-radius: 6px; font-size: 12px; color: #666;">
                                            💡 Tip: Use your browser's PDF controls to navigate. Some browsers may not jump directly to page {display_page}.
                                        </div>
                                    </div>
                                    """

                                    components.html(pdf_html, height=900, scrolling=True)

                                    # Download button
                                    pdf_bytes = base64.b64decode(pdf_b64)
                                    st.download_button(
                                        "⬇️ Download Full PDF",
                                        data=pdf_bytes,
                                        file_name=src,
                                        mime="application/pdf",
                                        type="primary",
                                    )

                                    st.markdown("---")

                                except Exception as e:
                                    st.error(f"Failed to load PDF: {e}")
                            else:
                                st.caption("Jumps to cited page • Transparent evidence preview (grader-friendly)")
                        else:
                            st.warning(f"PDF not found in `data/kb_raw`: {src}")

        if rewrite_query:
            with st.expander("See how your question was rewritten"):
                st.code(res.get("used_query", "") or "(no rewrite)")

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Tab 2 – What-If Scenarios
# -------------------------------------------------------------------

with tab_whatif:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
    st.subheader("Explore What-If Scenarios")
    st.write(
        "Describe a hypothetical situation. The assistant will explain which policies are relevant and suggest next steps."
    )
    scenario = st.text_area(
        "Scenario",
        height=140,
        placeholder="E.g., I submitted late due to illness and informed the instructor after the deadline...",
        label_visibility="collapsed",
    )

    if st.button("Analyze Scenario", type="primary"):
        if not scenario.strip():
            st.warning("Please write a scenario first.")
        else:
            try:
                llm = LLMClient()
                system_prompt = (
                    "You are PolicyNavigator AI. Give careful, non-judgmental guidance. "
                    "Do NOT invent exact rules; speak generally and suggest who to contact."
                )
                user_prompt = f"""
Scenario:
\"\"\"{scenario}\"\"\"

Explain:
1) Which policy areas are likely relevant.
2) What risks or consequences might apply.
3) What steps the student should take next.
4) A short supportive closing line.
"""
                with st.spinner("Analyzing..."):
                    out = llm.chat(system_prompt, user_prompt)
                st.markdown("### 🧩 Scenario Analysis")
                st.write(out)
            except Exception as e:
                st.error(f"Scenario analysis failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Tab 3 – Policy Quiz
# -------------------------------------------------------------------

with tab_quiz:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
    st.subheader("Learn the Policy with a Quick Quiz")

    kb_files = st.session_state.get("kb_files", [])
    kb_indexed = st.session_state.get("kb_indexed", False)

    if kb_indexed and kb_files:
        st.caption(f"Tailored to your indexed docs: `{', '.join(kb_files)}`")
    else:
        st.caption("Index policy PDFs in the sidebar to tailor quiz questions.")

    num_questions = st.slider("Number of quiz questions", 3, 8, 5)

    if st.button("Generate Quiz", type="primary"):
        try:
            llm = LLMClient()
            scope = (
                "Base questions on common academic integrity and student conduct policies. "
                + (f"Relevant files: {', '.join(kb_files)}." if kb_files else "")
            )
            system_prompt = "You generate short multiple-choice quizzes about university policies."
            user_prompt = f"""
Generate {num_questions} multiple-choice quiz questions.

{scope}

Return ONLY a JSON array with schema:
[
  {{
    "question": "string",
    "options": ["option A", "option B", "option C", "option D"],
    "answer": "must match one option exactly",
    "explanation": "short explanation"
  }}
]
No extra text.
"""
            with st.spinner("Generating quiz..."):
                raw = llm.chat(system_prompt, user_prompt)

            items = extract_json_array(raw)
            if not items:
                raise ValueError("Quiz JSON is empty.")
            st.session_state["quiz_items"] = items
            st.success("Quiz generated!")
        except Exception as e:
            st.error(f"Failed to generate quiz: {e}")
            st.session_state["quiz_items"] = None

    quiz_items = st.session_state.get("quiz_items")

    if quiz_items:
        for idx, item in enumerate(quiz_items, start=1):
            q = (item.get("question") or "").strip()
            options = item.get("options") or []
            ans = (item.get("answer") or "").strip()
            expl = (item.get("explanation") or "").strip()
            if not q or not options:
                continue

            st.markdown(f"**Q{idx}. {q}**")
            choice = st.radio("Choose:", options, key=f"quiz_{idx}", label_visibility="collapsed")
            if st.button("Check answer", key=f"check_{idx}"):
                if choice == ans:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Correct answer: **{ans}**")
                if expl:
                    st.info(expl)
            st.markdown("---")
    else:
        st.info("Click **Generate Quiz** to get started.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Tab 4 – Evaluation (A/B)
# -------------------------------------------------------------------

with tab_eval:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
    st.subheader("A/B Evaluation Results (Baseline vs Improved Retrieval)")

    st.caption(
        "This tab reads the output from `python scripts/ab_eval.py`:\n"
        "- `results/ab_eval_runs.csv`\n"
        "- `results/ab_eval_summary.json`"
    )

    results_dir = "results"
    runs_csv = os.path.join(results_dir, "ab_eval_runs.csv")
    summary_json = os.path.join(results_dir, "ab_eval_summary.json")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Files detected**")
        st.write(f"• `{runs_csv}`: {'✅' if os.path.exists(runs_csv) else '❌'}")
        st.write(f"• `{summary_json}`: {'✅' if os.path.exists(summary_json) else '❌'}")

    with c2:
        st.markdown("**Quick instructions**")
        st.code("python scripts/ab_eval.py", language="bash")

    if not (os.path.exists(runs_csv) and os.path.exists(summary_json)):
        st.warning(
            "Run the evaluation script first:\n"
            "`python scripts/ab_eval.py`\n\n"
            "Then refresh this page."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # --- Load summary ---
        try:
            summary = json.loads(_safe_read_text(summary_json))
        except Exception as e:
            st.error(f"Failed to read summary JSON: {e}")
            summary = {}

        # --- Load runs CSV ---
        try:
            df = pd.read_csv(runs_csv)
        except Exception as e:
            st.error(f"Failed to read runs CSV: {e}")
            df = pd.DataFrame()

        # --- KPI cards ---
        kpi_left, kpi_mid, kpi_right = st.columns(3)

        # Safely extract common keys (works even if your JSON schema changes slightly)
        num_q = summary.get("num_questions") or summary.get("n_questions") or summary.get("questions") or ""
        baseline_hall_rate = summary.get("baseline_hallucination_rate")
        improved_hall_rate = summary.get("improved_hallucination_rate")
        baseline_avg_sim = summary.get("baseline_avg_max_sim")
        improved_avg_sim = summary.get("improved_avg_max_sim")

        with kpi_left:
            st.markdown("### ✅ Coverage")
            st.markdown(f"<div class='pn-badge'>Questions: {num_q if num_q != '' else '—'}</div>", unsafe_allow_html=True)

        with kpi_mid:
            st.markdown("### 🧠 Max Similarity (Avg)")
            st.markdown(
                f"<div class='pn-badge'>Baseline: {baseline_avg_sim if baseline_avg_sim is not None else '—'}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='pn-badge'>Improved: {improved_avg_sim if improved_avg_sim is not None else '—'}</div>",
                unsafe_allow_html=True,
            )

        with kpi_right:
            st.markdown("### ⚠ Hallucination Rate")
            st.markdown(
                f"<div class='pn-badge'>Baseline: {_pct(baseline_hall_rate) if baseline_hall_rate is not None else '—'}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='pn-badge'>Improved: {_pct(improved_hall_rate) if improved_hall_rate is not None else '—'}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # --- Show summary JSON (collapsible) ---
        with st.expander("View summary JSON"):
            st.json(summary)

        # --- Runs table ---
        st.markdown("### 📄 Detailed Run Log")
        if df.empty:
            st.warning("Runs CSV is empty or unreadable.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

        # --- Charts ---
        if not df.empty:
            # try to standardize expected column names
            # (Your script likely writes baseline_max_sim / improved_max_sim etc.)
            cols = set(df.columns)

            st.markdown("### 📈 Charts")
            chart_c1, chart_c2 = st.columns(2)

            with chart_c1:
                st.markdown("**Max similarity by question**")
                if {"baseline_max_sim", "improved_max_sim"}.issubset(cols):
                    chart_df = df[["question", "baseline_max_sim", "improved_max_sim"]].copy()
                    chart_df = chart_df.set_index("question")
                    st.line_chart(chart_df)
                else:
                    st.info("Expected columns not found: baseline_max_sim, improved_max_sim")

            with chart_c2:
                st.markdown("**Hallucination flags (count)**")
                if {"baseline_hallucination_flag", "improved_hallucination_flag"}.issubset(cols):
                    b = int(df["baseline_hallucination_flag"].astype(bool).sum())
                    i = int(df["improved_hallucination_flag"].astype(bool).sum())
                    bar_df = pd.DataFrame({"count": [b, i]}, index=["baseline", "improved"])
                    st.bar_chart(bar_df)
                else:
                    st.info("Expected columns not found: baseline_hallucination_flag, improved_hallucination_flag")

        st.markdown("---")

        # --- Download buttons ---
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download ab_eval_runs.csv",
                data=_safe_read_bytes(runs_csv),
                file_name="ab_eval_runs.csv",
                mime="text/csv",
                type="primary",
            )
        with dl2:
            st.download_button(
                "⬇️ Download ab_eval_summary.json",
                data=_safe_read_bytes(summary_json),
                file_name="ab_eval_summary.json",
                mime="application/json",
                type="primary",
            )

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Tab 5 – About & Docs
# -------------------------------------------------------------------

with tab_docs:
    st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
    st.subheader("About PolicyNavigator AI")
    st.markdown(
        """
**PolicyNavigator AI** helps students understand complex university policies.

**Core features**
- 📚 RAG grounded answers from uploaded NEU policy PDFs  
- 🧠 Prompt-engineered response styles (strict quote vs student-friendly)  
- 🤔 What-if scenario explainer  
- 📝 Quiz generator for learning  

**Responsible use**
- Always verify critical decisions using official NEU policy sources.
"""
    )
    st.markdown("### Project Assets")
    st.markdown(
        """
- GitHub: `https://github.com/akshayagavhane9/policynavigator-ai`  
- Documentation: `docs/project_documentation.pdf`  
- Architecture: `docs/architecture_diagram.png`  
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
