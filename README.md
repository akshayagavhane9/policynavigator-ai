# 📘 PolicyNavigator AI  
**Policy Assistant · Retrieval-Augmented Generation (RAG) · Prompt Engineering · Synthetic Data Generation · Evaluation**

PolicyNavigator AI is an end-to-end **policy intelligence system** that helps students understand complex university policies in **plain English**, while remaining **strictly grounded in official policy documents**.

It supports:
- Grounded Q&A with citations
- Student-friendly explanations of rights
- What-if scenario analysis
- Auto-generated policy quizzes
- Synthetic evaluation and A/B testing of retrieval strategies

> 🎓 Built as a **Prompt Engineering final project** at Northeastern University, with a strong focus on **correctness, reproducibility, and evaluation**.

---

## ✨ Key Features

### 💬 Policy Q&A (RAG-Based)
- Natural-language questions over uploaded PDF/DOCX/TXT policy files
- Answers **strictly grounded** in retrieved document chunks
- Citations with similarity scores and metadata
- Query rewriting for improved retrieval

### 🧠 Prompt-Engineered Answer Styles
- **Strict policy quote** (verbatim, formal)
- **Student-friendly explanation** (rights & next steps)

### 🤔 What-If Scenario Reasoning
- Structured reasoning for hypothetical situations
- Non-judgmental, guidance-focused responses
- Designed for learning (not legal advice)

### 📝 Policy Quiz Generator
- Auto-generated multiple-choice questions
- Explanations for each answer
- Helps students actively learn policy content

### 🛡️ Hallucination Awareness
- Confidence scoring per answer
- Hallucination risk flagging
- Explicit "Not covered in policy" fallbacks

### 📊 Evaluation & Metrics (Top-25% Signal)
- Synthetic Q&A evaluation
- **A/B comparison: Baseline vs Improved RAG**
- Adaptive MMR vs pure similarity search
- Metrics logged to CSV + JSON for reproducibility

---

## 🧱 System Architecture

### High-Level
- Web UI (Landing Page)
- Streamlit Dashboard
- FastAPI Backend
- RAG Pipeline (Embedding → Retrieval → Generation)
- Vector Store
- LLM Provider
- Evaluation Pipeline

![Architecture Diagram](High-Level Architecture.png)

### Low-Level Highlights
- Adaptive MMR gating (disabled for small KBs)
- Chunk deduplication
- Query rewrite isolation
- Explicit citation construction
- Offline evaluation via scripts

(See `/Low-Level Architecture.png`)

---

## 🛠️ Tech Stack

**Backend & RAG**
- Python 3.10+
- FastAPI
- OpenAI API
- Vector similarity search
- Custom chunking & retrieval logic

**Frontend**
- Streamlit (interactive dashboard)
- HTML + CSS landing page (portfolio site)

**Evaluation**
- Synthetic Q&A datasets
- A/B evaluation scripts
- CSV / JSON metrics export

---

## 📁 Project Structure
```
policynavigator-ai/
├── src/
│   ├── main.py                # Core RAG logic
│   ├── llm/                   # LLM client
│   ├── rag/
│   │   ├── chunker.py         # Text chunking
│   │   ├── embeddings/        # Embedding logic
│   │   └── vectordb.py        # Similarity + MMR retrieval
│   └── ui/
│       └── streamlit_app.py   # Streamlit dashboard
├── scripts/
│   └── ab_eval.py             # A/B evaluation runner
├── data/
│   └── kb_raw/                # Uploaded policy files
├── results/
│   ├── ab_eval_runs.csv
│   └── ab_eval_summary.json
├── docs/
│   ├── project_documentation.pdf
│   └── High_level_architecture.png
    └── Low_level_architecture.png
├── index.html                 # Landing page
├── css/
│   └── styles.css
└── README.md
```

## 🚀 Setup Instructions (Reproducible)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/akshayagavhane9/policynavigator-ai.git
cd policynavigator-ai
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 5️⃣ Run Streamlit App
```bash
streamlit run src/ui/streamlit_app.py
```

### 6️⃣ (Optional) Run Evaluation
```bash
python scripts/ab_eval.py
```

Results will be saved in:
```
results/ab_eval_runs.csv
results/ab_eval_summary.json
```

---

## 📊 Evaluation Methodology

We explicitly compare:

| Mode | Description |
|------|-------------|
| **Baseline** | Top-K cosine similarity |
| **Improved** | Query rewrite + Adaptive MMR + deduplication |

**Metrics tracked:**
- Max similarity score
- Citation coverage
- Hallucination flag
- Retrieval robustness across questions

This makes improvements measurable, not anecdotal.

---

## 👥 Project Contributors

### - Ritwik Giri
### - Akshaya Gavhane


⚖️ **Both contributors collaborated across frontend, backend, prompting, and evaluation to ensure the system was not only functional, but explainable, measurable, and reproducible.**

---

## 🎯 Why This Project Stands Out
- Goes beyond "basic RAG"
- Explicit hallucination awareness
- Measurable improvements via A/B testing
- Clear separation of concerns
- Production-style error handling
- Portfolio-ready UI + documentation

---

## 📌 Disclaimer
PolicyNavigator AI is an educational assistant. Always verify important decisions with official university policy sources.
