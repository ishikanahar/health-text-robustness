import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.config import DATA_PROCESSED_DIR, TABLES_DIR


MODEL_NAME = "distilbert-base-uncased"


def load_transformer_data(input_filename="processed_symptom_to_diagnosis.csv"):
    input_path = DATA_PROCESSED_DIR / input_filename
    df = pd.read_csv(input_path)

    # keep same top 10 classes as LSTM
    top_labels = df["label"].value_counts().head(10).index
    df = df[df["label"].isin(top_labels)].copy()

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    unique_labels = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_df["label_id"] = train_df["label"].map(label_to_idx)
    test_df["label_id"] = test_df["label"].map(label_to_idx)

    return train_df, test_df, label_to_idx, idx_to_label


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train_transformer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_df, test_df, label_to_idx, idx_to_label = load_transformer_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )
    test_dataset = Dataset.from_pandas(
        test_df[["text", "label_id"]].rename(columns={"label_id": "label"})
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_to_idx),
        id2label=idx_to_label,
        label2id=label_to_idx,
    )

    training_args = TrainingArguments(
        output_dir=str(TABLES_DIR / "transformer_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    report = classification_report(
        labels,
        preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df),
        "num_classes": len(label_to_idx),
        "model_name": MODEL_NAME,
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = TABLES_DIR / "transformer_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    report_df = pd.DataFrame(report).transpose()
    report_path = TABLES_DIR / "transformer_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    model_dir = TABLES_DIR / "transformer_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    with open(TABLES_DIR / "transformer_label_to_idx.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)

    print("Final Transformer Results")
    print("-------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved classification report to: {report_path}")
    print(f"Saved model to: {model_dir}")


if __name__ == "__main__":
    train_transformer()
