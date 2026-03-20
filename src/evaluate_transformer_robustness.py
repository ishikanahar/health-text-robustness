import json
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.config import DATA_PROCESSED_DIR, TABLES_DIR
from src.perturb import synonym_replace, random_typo, drop_words


def load_transformer_eval_data(input_filename="processed_symptom_to_diagnosis.csv"):
    input_path = DATA_PROCESSED_DIR / input_filename
    df = pd.read_csv(input_path)

    top_labels = df["label"].value_counts().head(10).index
    df = df[df["label"].isin(top_labels)].copy()

    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    with open(TABLES_DIR / "transformer_label_to_idx.json", "r") as f:
        label_to_idx = json.load(f)

    idx_to_label = {int(v): k for k, v in label_to_idx.items()}

    test_df["label_id"] = test_df["label"].map(label_to_idx)

    return test_df, label_to_idx, idx_to_label


def make_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(
        df[["text", "label_id"]].rename(columns={"label_id": "label"})
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
    return dataset


def evaluate_transformer_robustness():
    model_dir = str(TABLES_DIR / "transformer_model")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    test_df, label_to_idx, idx_to_label = load_transformer_eval_data()

    scenarios = {
        "clean": test_df.copy(),
        "synonym": test_df.copy(),
        "typo": test_df.copy(),
        "drop": test_df.copy(),
    }

    scenarios["synonym"]["text"] = scenarios["synonym"]["text"].apply(synonym_replace)
    scenarios["typo"]["text"] = scenarios["typo"]["text"].apply(lambda x: random_typo(x, prob=0.1))
    scenarios["drop"]["text"] = scenarios["drop"]["text"].apply(lambda x: drop_words(x, prob=0.1))

    eval_args = TrainingArguments(
        output_dir=str(TABLES_DIR / "transformer_eval_tmp"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
    )

    results = {}

    for scenario_name, scenario_df in scenarios.items():
        dataset = make_dataset(scenario_df, tokenizer)
        predictions = trainer.predict(dataset)

        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")

        results[scenario_name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

    print("\nTransformer Robustness Results")
    print("------------------------------")
    for name, metrics in results.items():
        print(f"{name}: acc={metrics['accuracy']:.4f}, f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    evaluate_transformer_robustness()
