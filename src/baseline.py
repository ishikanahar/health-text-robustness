import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED_DIR, TABLES_DIR


def train_baseline(
    input_filename: str = "processed_symptom_to_diagnosis.csv"
):
    input_path = DATA_PROCESSED_DIR / input_filename
    df = pd.read_csv(input_path)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df),
        "num_classes": df["label"].nunique(),
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = TABLES_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    report_df = pd.DataFrame(report).transpose()
    report_path = TABLES_DIR / "baseline_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    print("\nBaseline Results")
    print("----------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved classification report to: {report_path}")

    return model, metrics, report_df


if __name__ == "__main__":
    train_baseline()
