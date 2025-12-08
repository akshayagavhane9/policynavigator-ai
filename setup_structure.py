import os

folders = [
    "src/api/routers",
    "src/api/models",
    "src/rag/loaders",
    "src/rag/preprocessors",
    "src/rag/embeddings",
    "src/rag/vectorstore",
    "src/rag/retriever",
    "src/llm/prompts",
    "src/evaluation",
    "src/ui/components",
    "src/utils",
    "tests",
    "data/kb_raw",
    "data/kb_processed",
    "data/synthetic_eval",
    "notebooks",
    "docs/images",
    "web/css",
    "web/js"
]

files = [
    "src/api/server.py",
    "src/rag/pipeline.py",
    "src/llm/client.py",
    "src/llm/prompt_builder.py",
    "src/main.py",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "CONTRIBUTING.md",
    ".env.example",
    "web/index.html"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, "w") as f:
        f.write("")

print("Repo structure created successfully!")
