import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.baseline import train_baseline
from src.config import DATA_PROCESSED_DIR
from src.perturb import synonym_replace, random_typo, drop_words


def load_data():
    df = pd.read_csv(DATA_PROCESSED_DIR / "processed_symptom_to_diagnosis.csv")
    top_labels = df["label"].value_counts().head(10).index
    df = df[df["label"].isin(top_labels)].copy()
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    return test_df


def evaluate():
    model, _, _ = train_baseline()
    test_df = load_data()

    scenarios = {
        "clean": test_df.copy(),
        "synonym": test_df.copy(),
        "typo": test_df.copy(),
        "drop": test_df.copy(),
    }

    scenarios["synonym"]["text"] = scenarios["synonym"]["text"].apply(synonym_replace)
    scenarios["typo"]["text"] = scenarios["typo"]["text"].apply(lambda x: random_typo(x, 0.1))
    scenarios["drop"]["text"] = scenarios["drop"]["text"].apply(lambda x: drop_words(x, 0.1))

    print("\nTF-IDF Robustness Results")
    print("-------------------------")

    for name, scenario_df in scenarios.items():
        y_true = scenario_df["label"]
        y_pred = model.predict(scenario_df["text"])

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        print(f"{name}: acc={acc:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    evaluate()