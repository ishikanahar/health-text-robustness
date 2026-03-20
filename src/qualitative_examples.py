from pathlib import Path
import pandas as pd
import random

from src.baseline import train_baseline
from src.config import DATA_PROCESSED_DIR
from src.perturb import random_typo

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)


def load_test_data():
    df = pd.read_csv(DATA_PROCESSED_DIR / "processed_symptom_to_diagnosis.csv")
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    return test_df


def main():
    model, _, _ = train_baseline()
    test_df = load_test_data()

    # pick 10 random examples
    sample_df = test_df.sample(n=10, random_state=42).copy()

    rows = []

    for _, row in sample_df.iterrows():
        original_text = row["text"]
        true_label = row["label"]

        typo_text = random_typo(original_text, prob=0.1)

        pred_clean = model.predict([original_text])[0]
        pred_typo = model.predict([typo_text])[0]

        rows.append({
            "Original Text": original_text,
            "Typo Text": typo_text,
            "True Label": true_label,
            "Prediction Clean": pred_clean,
            "Prediction Typo": pred_typo
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(results_dir / "qualitative_examples.csv", index=False)

    print("Saved qualitative examples to results/")


if __name__ == "__main__":
    main()
