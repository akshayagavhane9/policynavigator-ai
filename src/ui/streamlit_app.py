import streamlit as st
from src.llm.prompt_builder import PromptBuilder
from src.llm.client import LLMClient

st.set_page_config(page_title="PolicyNavigator AI", layout="wide")
st.title("📘 PolicyNavigator AI")

prompt_builder = PromptBuilder()
llm = LLMClient()

st.subheader("Ask a Policy Question")

question = st.text_input("Your question:")

if st.button("Generate"):
    rewritten = llm.run(prompt_builder.build_rewrite_prompt(question))
    st.write("🔍 Search-Optimized Query:", rewritten)

    st.info("Backend RAG retrieval not connected yet — using test prompt only.")
    answer = llm.run(prompt_builder.build_answer_prompt(question, "TEST_CONTEXT"))
    st.json(answer)
