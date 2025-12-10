import sys
import os

# Add the project root directory to PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import streamlit as st

from src.main import answer_question
from src.rag.pipeline import index_documents


st.set_page_config(page_title="PolicyNavigator AI", layout="wide")

st.title("📘 PolicyNavigator AI")
st.caption("Ask questions grounded in your own policy documents.")


# Sidebar: document upload & indexing
st.sidebar.header("📂 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload policy documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

index_clicked = st.sidebar.button("Index Documents")

if index_clicked:
    if uploaded_files:
        import os

        raw_dir = os.path.join("data", "kb_raw")
        os.makedirs(raw_dir, exist_ok=True)

        file_paths = []
        for f in uploaded_files:
            save_path = os.path.join(raw_dir, f.name)
            with open(save_path, "wb") as out:
                out.write(f.read())
            file_paths.append(save_path)

        with st.spinner("Indexing documents... this may take a moment."):
            index_documents(file_paths)

        st.sidebar.success(f"Indexed {len(file_paths)} document(s).")
    else:
        st.sidebar.warning("Please upload at least one file before indexing.")


# Main: Q&A interface
st.subheader("🔍 Ask a Policy Question")

question = st.text_input("Enter your question:")

col1, col2 = st.columns([1, 3])

with col1:
    use_rewrite = st.checkbox("Rewrite query", value=True)

with col2:
    top_k = st.slider("Number of context chunks to retrieve", min_value=3, max_value=10, value=5)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = answer_question(
                question=question,
                use_rewrite=use_rewrite,
                top_k=top_k,
                collection_name="policies",
            )

        st.markdown("### 🧠 Answer")
        st.write(result.get("answer", ""))

        st.markdown("### 📎 Citations")
        citations = result.get("citations", [])
        if citations:
            for c in citations:
                st.write(f"- {c}")
        else:
            st.write("_No citations returned._")

        st.markdown("### 📊 Confidence")
        st.write(result.get("confidence", "unknown"))
