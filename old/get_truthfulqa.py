from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path("data")
OUT.mkdir(exist_ok=True)


def main():
    ds = load_dataset("truthful_qa", "generation")
    val = ds["validation"]
    df = pd.DataFrame(
        {
            "question": val["question"],
            "best_answer": val["best_answer"],
            "category": val["category"],
        }
    )
    df.to_csv(OUT / "truthfulqa_generation_full.csv", index=False, encoding="utf-8")
    df.sample(n=min(50, len(df)), random_state=42).to_csv(
        OUT / "truthfulqa_generation_50.csv", index=False, encoding="utf-8"
    )
    print(
        "Saved:",
        OUT / "truthfulqa_generation_full.csv",
        "and",
        OUT / "truthfulqa_generation_50.csv",
    )


if __name__ == "__main__":
    main()
