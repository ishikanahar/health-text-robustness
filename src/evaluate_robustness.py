import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.baseline import train_baseline
from src.perturb import synonym_replace, random_typo, drop_words
from src.config import DATA_PROCESSED_DIR


def evaluate():
    model, _, _ = train_baseline()

    df = pd.read_csv(DATA_PROCESSED_DIR / "processed_symptom_to_diagnosis.csv")
    test_df = df[df["split"] == "test"].copy()

    X = test_df["text"]
    y = test_df["label"]

    scenarios = {
        "clean": X,
        "synonym": X.apply(synonym_replace),
        "typo": X.apply(random_typo),
        "drop": X.apply(drop_words),
    }

    results = {}

    for name, X_mod in scenarios.items():
        preds = model.predict(X_mod)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")

        results[name] = {
            "accuracy": acc,
            "macro_f1": f1
        }

    print("\nRobustness Results")
    print("------------------")
    for k, v in results.items():
        print(f"{k}: acc={v['accuracy']:.4f}, f1={v['macro_f1']:.4f}")

    return results


if __name__ == "__main__":
    evaluate()
