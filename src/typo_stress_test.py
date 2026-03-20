import pandas as pd
from sklearn.metrics import accuracy_score
from src.baseline import train_baseline
from src.config import DATA_PROCESSED_DIR
from src.perturb import random_typo

def load_data():
    df = pd.read_csv(DATA_PROCESSED_DIR / "processed_symptom_to_diagnosis.csv")
    top_labels = df["label"].value_counts().head(10).index
    df = df[df["label"].isin(top_labels)]
    return df[df["split"] == "test"].copy()

def run():
    model, _, _ = train_baseline()
    df = load_data()

    probs = [0.05, 0.1, 0.2, 0.3]

    print("\nTF-IDF Typo Stress Test")
    print("----------------------")

    for p in probs:
        noisy = df.copy()
        noisy["text"] = noisy["text"].apply(lambda x: random_typo(x, p))

        y_true = noisy["label"]
        y_pred = model.predict(noisy["text"])

        acc = accuracy_score(y_true, y_pred)
        print(f"prob={p}: acc={acc:.4f}")

if __name__ == "__main__":
    run()
