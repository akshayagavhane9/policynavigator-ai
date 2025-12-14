# scripts/ab_eval_plot.py
import os
import pandas as pd
import matplotlib.pyplot as plt

RUNS_CSV = "results/ab_eval_runs.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    if not os.path.exists(RUNS_CSV):
        raise FileNotFoundError(f"Not found: {RUNS_CSV}. Run `python scripts/ab_eval.py` first.")

    df = pd.read_csv(RUNS_CSV)

    # --- auto-detect schema ---
    question_col = pick_col(df, ["question", "q", "prompt"])
    variant_col = pick_col(df, ["variant", "arm", "mode", "run_type", "setting"])
    sim_col = pick_col(df, ["max_sim", "max_similarity", "confidence_score", "max_score", "top_score", "score"])
    citations_col = pick_col(df, ["citations", "num_citations", "citation_count"])
    hall_col = pick_col(df, ["hallucination_flag", "hall_flag", "hall", "is_hallucinated"])

    missing = []
    if not question_col: missing.append("question")
    if not variant_col: missing.append("variant")
    if not sim_col: missing.append("max_sim/confidence_score")

    if missing:
        raise KeyError(
            f"CSV columns missing/unknown: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Fix: rename your CSV columns OR tell me the header line."
        )

    # normalize names for plotting
    df = df.rename(
        columns={
            question_col: "question",
            variant_col: "variant",
            sim_col: "sim",
        }
    )

    if citations_col:
        df = df.rename(columns={citations_col: "citations"})
    if hall_col:
        df = df.rename(columns={hall_col: "hall"})
    else:
        df["hall"] = False

    # clean / types
    df["question"] = df["question"].astype(str)
    df["variant"] = df["variant"].astype(str)
    df["sim"] = pd.to_numeric(df["sim"], errors="coerce").fillna(0.0)

    def short_q(x: str) -> str:
        x = " ".join(x.strip().split())
        return x[:55] + ("…" if len(x) > 55 else "")

    df["question_short"] = df["question"].apply(short_q)

    # hall as bool
    df["hall"] = df["hall"].astype(str).str.lower().isin(["true", "1", "yes", "y"])

    # -------------------------------
    # Chart 1: Similarity by question (baseline vs improved)
    # -------------------------------
    pivot_sim = df.pivot_table(
        index="question_short",
        columns="variant",
        values="sim",
        aggfunc="mean",
    )

    ax = pivot_sim.plot(kind="bar", figsize=(12, 6))
    ax.set_title("A/B Retrieval Quality: Similarity by Question")
    ax.set_xlabel("")
    ax.set_ylabel("Similarity (cosine)")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, "ab_eval_similarity.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    # -------------------------------
    # Chart 2: Hallucination flags count
    # -------------------------------
    hall_counts = df.groupby("variant")["hall"].sum()

    plt.figure(figsize=(6, 5))
    hall_counts.plot(kind="bar")
    plt.title("Hallucination Flags Count (Lower is Better)")
    plt.xlabel("")
    plt.ylabel("Flagged Answers")
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "ab_eval_hallucination_flags.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    # -------------------------------
    # Chart 3: Citations by question (optional)
    # -------------------------------
    if "citations" in df.columns:
        df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0)
        pivot_cit = df.pivot_table(
            index="question_short",
            columns="variant",
            values="citations",
            aggfunc="mean",
        )
        ax = pivot_cit.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Citations Returned per Question")
        ax.set_xlabel("")
        ax.set_ylabel("Citations")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out3 = os.path.join(OUT_DIR, "ab_eval_citations.png")
        plt.savefig(out3, dpi=200)
        plt.close()

        print("✅ Wrote:")
        print(" -", out1)
        print(" -", out2)
        print(" -", out3)
    else:
        print("✅ Wrote:")
        print(" -", out1)
        print(" -", out2)

    print("\nDetected columns:")
    print(" question:", question_col)
    print(" variant:", variant_col)
    print(" sim:", sim_col)
    print(" citations:", citations_col)
    print(" hall:", hall_col)


if __name__ == "__main__":
    main()
