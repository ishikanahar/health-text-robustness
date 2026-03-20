import json
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    DATA_PROCESSED_DIR,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text: str):
    return str(text).split()


def build_vocab(texts, max_vocab_size=MAX_VOCAB_SIZE):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    most_common_words = counter.most_common(max_vocab_size - 2)

    word_to_idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for idx, (word, _) in enumerate(most_common_words, start=2):
        word_to_idx[word] = idx

    return word_to_idx


def encode_text(text, word_to_idx, max_length=MAX_SEQUENCE_LENGTH):
    tokens = tokenize(text)
    token_ids = [word_to_idx.get(token, word_to_idx[UNK_TOKEN]) for token in tokens]

    if len(token_ids) < max_length:
        token_ids += [word_to_idx[PAD_TOKEN]] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    return token_ids


class SymptomDataset(Dataset):
    def __init__(self, df, word_to_idx, label_to_idx):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.word_to_idx = word_to_idx
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids = encode_text(text, self.word_to_idx)
        label_id = self.label_to_idx[label]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def load_lstm_data(input_filename="processed_symptom_to_diagnosis.csv"):
    input_path = DATA_PROCESSED_DIR / input_filename
    df = pd.read_csv(input_path)

    # 🔥 keep only top 10 classes
    top_labels = df["label"].value_counts().head(10).index
    df = df[df["label"].isin(top_labels)].copy()

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    word_to_idx = build_vocab(train_df["text"].tolist())

    unique_labels = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_dataset = SymptomDataset(train_df, word_to_idx, label_to_idx)
    test_dataset = SymptomDataset(test_df, word_to_idx, label_to_idx)

    return train_df, test_df, train_dataset, test_dataset, word_to_idx, label_to_idx, idx_to_label


def save_mappings(word_to_idx, label_to_idx):
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PROCESSED_DIR / "word_to_idx.json", "w") as f:
        json.dump(word_to_idx, f, indent=2)

    with open(DATA_PROCESSED_DIR / "label_to_idx.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)


if __name__ == "__main__":
    train_df, test_df, train_dataset, test_dataset, word_to_idx, label_to_idx, idx_to_label = load_lstm_data()
    save_mappings(word_to_idx, label_to_idx)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Vocab size: {len(word_to_idx)}")
    print(f"Num classes: {len(label_to_idx)}")
