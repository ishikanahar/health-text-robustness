from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.baseline import train_baseline
from src.config import DATA_PROCESSED_DIR, TABLES_DIR
from src.perturb import random_typo

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)


def load_top10_test_df():
    df = pd.read_csv(DATA_PROCESSED_DIR / "processed_symptom_to_diagnosis.csv")
    top_labels = df["label"].value_counts().head(10).index
    df = df[df["label"].isin(top_labels)].copy()
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    return test_df


def plot_tfidf_clean_confusion():
    model, _, _ = train_baseline()
    test_df = load_top10_test_df()

    y_true = test_df["label"]
    y_pred = model.predict(test_df["text"])

    labels = sorted(test_df["label"].unique())

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        xticks_rotation=90,
        ax=ax,
        colorbar=False
    )
    plt.title("TF-IDF Confusion Matrix on Clean Data")
    plt.tight_layout()
    plt.savefig(results_dir / "tfidf_clean_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_transformer_typo_confusion():
    model_dir = str(TABLES_DIR / "transformer_model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    test_df = load_top10_test_df()

    with open(TABLES_DIR / "transformer_label_to_idx.json", "r") as f:
        label_to_idx = json.load(f)

    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    test_df["label_id"] = test_df["label"].map(label_to_idx)

    typo_df = test_df.copy()
    typo_df["text"] = typo_df["text"].apply(lambda x: random_typo(x, prob=0.1))

    dataset = Dataset.from_pandas(
        typo_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    eval_args = TrainingArguments(
        output_dir=str(TABLES_DIR / "transformer_eval_tmp_confusion"),
        report_to="none",
    )

    trainer = Trainer(model=model, args=eval_args)
    predictions = trainer.predict(dataset)

    preds = np.argmax(predictions.predictions, axis=1)
    labels_true = predictions.label_ids

    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        labels_true,
        preds,
        labels=list(range(len(target_names))),
        display_labels=target_names,
        xticks_rotation=90,
        ax=ax,
        colorbar=False
    )
    plt.title("Transformer Confusion Matrix on Typo Perturbation")
    plt.tight_layout()
    plt.savefig(results_dir / "transformer_typo_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_tfidf_clean_confusion()
    plot_transformer_typo_confusion()
    print("Saved confusion matrices to results/")
