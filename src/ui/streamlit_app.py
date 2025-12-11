import os
import sys
import json
import re
from pathlib import Path
from typing import List

# Ensure project root is on sys.path so `src` imports work in Streamlit
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.main import answer_question, ingest_and_index_documents  # type: ignore
from src.llm.client import LLMClient  # type: ignore


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def extract_json_array(raw: str) -> list:
    """
    Extract a JSON array from an LLM response that may contain extra text.
    Looks for the first '[' and the last ']' and tries to json.loads that slice.
    """
    if not raw or not raw.strip():
        raise ValueError("Model returned an empty response.")

    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        raise ValueError("Model did not return a JSON array.")
    json_str = match.group(0)
    return json.loads(json_str)


def confidence_color(label: str) -> str:
    label = (label or "").lower()
    if label == "high":
        return "✅ High"
    if label == "medium":
        return "🟠 Medium"
    return "⚪ Low"


# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="PolicyNavigator AI",
    page_icon="📘",
    layout="wide",
)

# -------------------------------------------------------------------
# Sidebar – Knowledge Base + Links
# -------------------------------------------------------------------

st.sidebar.title("📚 Knowledge Base")
st.sidebar.caption(
    "Upload NEU policy documents (PDF, DOCX, TXT). "
    "They will be used to ground answers."
)

uploaded_files = st.sidebar.file_uploader(
    "Drag and drop files here",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt"],
    label_visibility="collapsed",
)

if uploaded_files:
    for f in uploaded_files:
        st.sidebar.write(f"• {f.name}")

if st.sidebar.button("Index Documents", type="primary"):
    try:
        raw_dir = "data/kb_raw"
        os.makedirs(raw_dir, exist_ok=True)
        saved_paths: List[str] = []

        if not uploaded_files:
            st.sidebar.warning("Please upload at least one file first.")
        else:
            for uf in uploaded_files:
                save_path = os.path.join(raw_dir, uf.name)
                # Write/overwrite file contents
                with open(save_path, "wb") as out:
                    out.write(uf.read())
                saved_paths.append(save_path)

            if saved_paths:
                n_chunks = ingest_and_index_documents(saved_paths)
                st.sidebar.success(
                    f"Indexed {n_chunks} chunks from {len(saved_paths)} file(s)."
                )
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
- What is the late submission penalty?
- How are academic integrity violations handled?
- What is the appeal process for grade disputes?
"""
)

# -------------------------------------------------------------------
# Main Header
# -------------------------------------------------------------------

st.markdown(
    """
<div style="display:flex; align-items:center; gap:16px;">
    <div style="font-size:48px;">📘</div>
    <div>
        <h1 style="margin-bottom:0;">PolicyNavigator AI</h1>
        <p style="margin-top:4px; color:#555;">
            A retrieval-augmented assistant that answers university policy questions, explains student rights,
            simulates real-world scenarios, and even quizzes you.
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs: Q&A, What-if, Quiz, About
tab_qa, tab_whatif, tab_quiz, tab_docs = st.tabs(
    ["💬 Q&A Assistant", "🤔 What-If Scenarios", "📝 Policy Quiz", "📘 About & Docs"]
)

# -------------------------------------------------------------------
# Tab 1 – Q&A Assistant
# -------------------------------------------------------------------

with tab_qa:
    st.subheader("Ask a Policy Question")

    col_style, col_k = st.columns([2, 1])
    with col_style:
        answer_style = st.radio(
            "Answer style",
            ["Strict policy quote", "Explain my rights (student-friendly)"],
            horizontal=False,
        )
    with col_k:
        k = st.slider("Context chunks", min_value=3, max_value=10, value=5)
        rewrite_query = st.checkbox(
            "Rewrite query for retrieval", value=True, help="Improves retrieval quality"
        )

    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the policy on cheating in exams?",
    )

    if st.button("Generate Answer", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Consulting your policy documents..."):
                try:
                    res = answer_question(
                        user_question,
                        answer_style=answer_style,
                        rewrite_query=rewrite_query,
                        k=k,
                    )
                except Exception as e:
                    st.error(f"Something went wrong while generating the answer: {e}")
                    res = None

        if res:
            col_ans, col_meta = st.columns([2, 1])

            with col_ans:
                st.markdown("### 🧠 Answer")
                st.write(res.get("answer", ""))

            with col_meta:
                st.markdown("### 📊 Confidence & Metrics")
                label = res.get("confidence_label", "low")
                score = res.get("confidence_score", 0.0)
                st.write("**Confidence**")
                st.markdown(
                    f"""
<div style="padding:12px 16px; border-radius:12px;
            background:#FFF3E0; border:1px solid #FFB74D;">
    <div style="font-size:22px; font-weight:600;">{confidence_color(label)}</div>
    <div style="color:#555; margin-top:4px;">Score: {score:.2f}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                latency = res.get("latency_ms", 0)
                st.caption(f"Latency: {latency} ms")

                st.markdown("### ⭐ Feedback")
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    st.button("👍 Helpful", key="fb_helpful")
                with col_fb2:
                    st.button("👎 Not helpful", key="fb_not_helpful")

            st.markdown("### 📎 Citations")
            citations = res.get("citations", []) or []
            if not citations:
                st.write("No citations returned by the system.")
            else:
                for c in citations:
                    src = c.get("source", "unknown")
                    chunk_id = c.get("chunk_id", "")
                    rank = c.get("rank", "")
                    st.markdown(f"- **{src}** – chunk `{chunk_id}` (rank {rank})")

            if rewrite_query:
                used_query = res.get("used_query", "")
                with st.expander("See how your question was rewritten"):
                    st.code(used_query or "(no rewrite used)", language="text")

# -------------------------------------------------------------------
# Tab 2 – What-If Scenarios
# -------------------------------------------------------------------

with tab_whatif:
    st.subheader("Explore What-If Scenarios")

    st.write(
        "Describe a hypothetical situation (e.g., *I submitted the assignment 3 days late "
        "and my teammate copied part of my work*). The assistant will reason about how policies "
        "might apply — not as legal advice, but as an educational explanation."
    )

    scenario = st.text_area(
        "Describe your scenario",
        height=140,
        placeholder="E.g., I missed an exam due to illness and only informed the instructor afterwards...",
    )

    if st.button("Analyze Scenario"):
        if not scenario.strip():
            st.warning("Please describe a scenario first.")
        else:
            try:
                llm = LLMClient()
                system_prompt = (
                    "You are PolicyNavigator AI. You reason about university policies, "
                    "academic integrity, and student conduct in a careful, non-judgmental way. "
                    "You *do not* invent rules; you explain typical outcomes and questions a "
                    "student should ask their advisor or instructor."
                )

                user_prompt = f"""
Here is a student's scenario:

\"\"\"{scenario}\"\"\"

Using general university policy principles, explain:

1. Which kinds of policies are probably relevant.
2. What risks or consequences might apply.
3. What steps the student should take next (e.g., who to contact, how to document things).
4. A short, encouraging closing note.
"""

                with st.spinner("Thinking through your scenario..."):
                    answer = llm.chat(system_prompt, user_prompt)

                st.markdown("### 🧩 Scenario Analysis")
                st.write(answer)
            except Exception as e:
                st.error(f"Failed to analyze scenario: {e}")

# -------------------------------------------------------------------
# Tab 3 – Policy Quiz
# -------------------------------------------------------------------

with tab_quiz:
    st.subheader("Learn the Policy with a Quick Quiz")
    st.caption(
        "The quiz is generated from general university policy knowledge using the same LLM. "
        "It’s a lightweight learning tool — not an official assessment."
    )

    num_questions = st.slider(
        "Number of quiz questions", min_value=3, max_value=8, value=5
    )

    if st.button("Generate Quiz"):
        try:
            llm = LLMClient()

            system_prompt = (
                "You are an expert tutor on university academic integrity and student conduct policies. "
                "You generate short quizzes to help students understand key rules, violations, and consequences."
            )

            user_prompt = f"""
Generate {num_questions} multiple-choice quiz questions about university academic integrity
and student conduct policies (for example: cheating, plagiarism, collaboration rules,
sanctions, appeals).

Return your answer ONLY as a JSON array (no prose before or after) with this exact schema:

[
  {{
    "question": "string",
    "options": ["option A", "option B", "option C", "option D"],
    "answer": "the correct option text (must match one element of 'options')",
    "explanation": "short explanation of why this is the correct answer"
  }},
  ...
]

Do not include any keys other than question, options, answer, explanation.
If you are unsure of a detail, ask a more general question instead of inventing rules.
"""

            with st.spinner("Generating quiz questions..."):
                raw = llm.run(system_prompt, user_prompt)

            quiz_items = extract_json_array(raw)

            if not isinstance(quiz_items, list) or not quiz_items:
                raise ValueError("Parsed quiz is empty or not a list.")

            st.success("Quiz generated! Scroll down to practice.")

            for idx, item in enumerate(quiz_items, start=1):
                question = (item.get("question") or "").strip()
                options = item.get("options") or []
                answer = (item.get("answer") or "").strip()
                explanation = (item.get("explanation") or "").strip()

                if not question or not options:
                    continue

                with st.container():
                    st.markdown(f"**Q{idx}. {question}**")

                    user_choice = st.radio(
                        "Your answer:",
                        options,
                        key=f"quiz_q_{idx}",
                        label_visibility="collapsed",
                    )

                    if st.button("Check answer", key=f"quiz_check_{idx}"):
                        if user_choice == answer:
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Not quite. Correct answer: **{answer}**")

                        if explanation:
                            st.info(explanation)

                    st.markdown("---")

        except Exception as e:
            st.error(f"Failed to generate quiz: {e}")

    else:
        st.info("Click **Generate Quiz** to get started.")

# -------------------------------------------------------------------
# Tab 4 – About & Docs
# -------------------------------------------------------------------

with tab_docs:
    st.subheader("About PolicyNavigator AI")

    st.markdown(
        """
**PolicyNavigator AI** is your assistant for understanding complex university policies.

It combines:

- 🔧 **Prompt Engineering** – structured prompts for Q&A, what-if reasoning, and quiz generation  
- 📚 **RAG (Retrieval-Augmented Generation)** – answers grounded in your uploaded policy PDFs  
- 🤖 **LLM-based Reasoning & Tutoring** – scenario analysis and practice questions  

This interface is for educational use only and does **not** replace official university policy documents or legal advice.
"""
    )

    st.markdown("### 📂 Project Assets")
    st.markdown(
        """
- **GitHub:** `https://github.com/akshayagavhane9/policynavigator-ai`  
- **Technical Documentation:** `docs/project_documentation.pdf`  
- **Architecture Diagram:** `docs/architecture_diagram.png`  
"""
    )

    st.markdown("### ⚠️ Ethical & Responsible Use")
    st.markdown(
        """
- Always double-check critical decisions against the official NEU policy documents.  
- Be mindful of bias: the model reflects its training data and may not perfectly match NEU rules.  
- Do not use this system to justify academic misconduct or to dispute official rulings on its own.  
"""
    )
