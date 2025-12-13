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


def render_pdf_modal_pdfjs(pdf_b64: str, title: str, page: int = 1, height_px: int = 780) -> None:
    """
    Reliable "modal-like" PDF preview using PDF.js.
    - No file:// iframes (which browsers block)
    - Supports initial page jump
    """
    page = max(1, int(page or 1))

    # NOTE: Using CDN for pdf.js is simplest for demo. If you want offline, we can bundle it locally.
    html = f"""
    <style>
      .pn-overlay {{
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.62);
        z-index: 9998;
      }}
      .pn-modal {{
        position: fixed;
        top: 5%;
        left: 5%;
        width: 90%;
        height: 90%;
        background: #ffffff;
        z-index: 9999;
        border-radius: 16px;
        box-shadow: 0 18px 60px rgba(0,0,0,0.45);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }}
      .pn-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 14px;
        background: linear-gradient(90deg, #111 0%, #222 70%, #ff7a00 140%);
        color: #fff;
        border-bottom: 1px solid rgba(255,255,255,0.12);
      }}
      .pn-title {{
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.2px;
      }}
      .pn-meta {{
        font-size: 12px;
        opacity: 0.92;
      }}
      .pn-body {{
        flex: 1;
        background: #f6f7f9;
        padding: 10px;
        overflow: auto;
      }}
      .pn-toolbar {{
        display: flex;
        gap: 10px;
        align-items: center;
        padding: 8px 10px;
        background: #fff;
        border: 1px solid #eee;
        border-radius: 12px;
        margin-bottom: 10px;
      }}
      .pn-btn {{
        border: 1px solid #e6e6e6;
        background: #fff;
        padding: 6px 10px;
        border-radius: 10px;
        cursor: pointer;
        font-size: 12px;
      }}
      .pn-btn:hover {{ background: #f5f5f5; }}
      .pn-input {{
        width: 70px;
        padding: 6px 8px;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        font-size: 12px;
      }}
      canvas {{
        background: #fff;
        border-radius: 12px;
        border: 1px solid #eee;
        box-shadow: 0 10px 24px rgba(0,0,0,0.06);
        display: block;
        margin: 0 auto;
      }}
      .pn-foot {{
        padding: 10px 14px;
        background: #fff;
        border-top: 1px solid #eee;
        font-size: 12px;
        color: #666;
        display: flex;
        justify-content: space-between;
      }}
    </style>

    <div class="pn-overlay"></div>
    <div class="pn-modal" role="dialog" aria-modal="true">
      <div class="pn-header">
        <div>
          <div class="pn-title">📄 {title}</div>
          <div class="pn-meta">PDF preview (PDF.js) • jump-to-page supported</div>
        </div>
        <div class="pn-meta">Requested page: {page}</div>
      </div>

      <div class="pn-body">
        <div class="pn-toolbar">
          <button class="pn-btn" onclick="prevPage()">◀ Prev</button>
          <button class="pn-btn" onclick="nextPage()">Next ▶</button>
          <span style="font-size:12px;color:#444;">Page</span>
          <input class="pn-input" id="pageNum" type="number" min="1" value="{page}" />
          <button class="pn-btn" onclick="goToPage()">Go</button>
          <span style="margin-left:auto;font-size:12px;color:#666;" id="pageInfo"></span>
        </div>

        <canvas id="the-canvas"></canvas>
      </div>

      <div class="pn-foot">
        <div>Tip: Use the page box to jump to the cited page.</div>
        <div>Close using the Streamlit “Close Preview” button.</div>
      </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script>
      const pdfData = atob("{pdf_b64}");
      const pdfBytes = new Uint8Array(pdfData.length);
      for (let i = 0; i < pdfData.length; i++) {{
        pdfBytes[i] = pdfData.charCodeAt(i);
      }}

      const loadingTask = pdfjsLib.getDocument({{ data: pdfBytes }});
      let pdfDoc = null;
      let pageNumber = {page};
      let pageRendering = false;
      let pageNumPending = null;
      const scale = 1.25;

      const canvas = document.getElementById('the-canvas');
      const ctx = canvas.getContext('2d');
      const pageInfo = document.getElementById('pageInfo');
      const pageNumInput = document.getElementById('pageNum');

      function renderPage(num) {{
        pageRendering = true;
        pdfDoc.getPage(num).then(function(page) {{
          const viewport = page.getViewport({{ scale: scale }});
          canvas.height = viewport.height;
          canvas.width = viewport.width;

          const renderContext = {{
            canvasContext: ctx,
            viewport: viewport
          }};
          const renderTask = page.render(renderContext);

          renderTask.promise.then(function() {{
            pageRendering = false;
            pageInfo.textContent = `of ${pdfDoc.numPages}`;
            pageNumInput.value = num;

            if (pageNumPending !== null) {{
              renderPage(pageNumPending);
              pageNumPending = null;
            }}
          }});
        }});
      }}

      function queueRenderPage(num) {{
        if (num < 1) num = 1;
        if (num > pdfDoc.numPages) num = pdfDoc.numPages;
        pageNumber = num;

        if (pageRendering) {{
          pageNumPending = num;
        }} else {{
          renderPage(num);
        }}
      }}

      function prevPage() {{
        if (!pdfDoc) return;
        if (pageNumber <= 1) return;
        queueRenderPage(pageNumber - 1);
      }}

      function nextPage() {{
        if (!pdfDoc) return;
        if (pageNumber >= pdfDoc.numPages) return;
        queueRenderPage(pageNumber + 1);
      }}

      function goToPage() {{
        if (!pdfDoc) return;
        const n = parseInt(pageNumInput.value || "1");
        queueRenderPage(n);
      }}

      loadingTask.promise.then(function(pdf) {{
        pdfDoc = pdf;
        pageInfo.textContent = `of ${pdfDoc.numPages}`;
        queueRenderPage(pageNumber);
      }});
    </script>
    """
    components.html(html, height=height_px)


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

# Modal preview state
if "pdf_modal_open" not in st.session_state:
    st.session_state["pdf_modal_open"] = False
if "pdf_modal_source" not in st.session_state:
    st.session_state["pdf_modal_source"] = None
if "pdf_modal_page" not in st.session_state:
    st.session_state["pdf_modal_page"] = 1
if "pdf_modal_b64" not in st.session_state:
    st.session_state["pdf_modal_b64"] = None


# -------------------------------------------------------------------
# Styling (professional, clean, non-cluttered)
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

tab_qa, tab_whatif, tab_quiz, tab_docs = st.tabs(
    ["💬 Q&A Assistant", "🤔 What-If Scenarios", "📝 Policy Quiz", "📘 About & Docs"]
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

    res: Optional[Dict[str, Any]] = None

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
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    res = None

    st.markdown("</div>", unsafe_allow_html=True)

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
                            c1, c2, c3 = st.columns([1.2, 1.0, 2.0])
                            with c1:
                                if st.button(
                                    f"📄 View PDF page",
                                    key=f"view_pdf_{src}_{chunk_id}_{rank}",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    st.session_state["pdf_modal_open"] = True
                                    st.session_state["pdf_modal_source"] = src
                                    st.session_state["pdf_modal_page"] = int(page) if page else 1
                                    st.session_state["pdf_modal_b64"] = read_pdf_as_base64(pdf_path)
                            with c2:
                                st.caption("Jumps to cited page")
                            with c3:
                                st.caption("Transparent evidence preview (grader-friendly)")
                        else:
                            st.warning(f"PDF not found in `data/kb_raw`: {src}")

        if rewrite_query:
            with st.expander("See how your question was rewritten"):
                st.code(res.get("used_query", "") or "(no rewrite)")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Modal Render (PDF.js) ---
    if st.session_state.get("pdf_modal_open") and st.session_state.get("pdf_modal_b64"):
        st.markdown("<div class='pn-card'>", unsafe_allow_html=True)
        st.markdown("### 📄 PDF Preview")
        st.caption("If you don’t see the PDF, your network may be blocking the PDF.js CDN. Try another browser.")

        render_pdf_modal_pdfjs(
            pdf_b64=st.session_state["pdf_modal_b64"],
            title=str(st.session_state.get("pdf_modal_source") or "Policy PDF"),
            page=int(st.session_state.get("pdf_modal_page") or 1),
            height_px=820,
        )

        if st.button("❌ Close Preview", type="primary"):
            st.session_state["pdf_modal_open"] = False
            st.session_state["pdf_modal_source"] = None
            st.session_state["pdf_modal_page"] = 1
            st.session_state["pdf_modal_b64"] = None

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
# Tab 4 – About & Docs
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
