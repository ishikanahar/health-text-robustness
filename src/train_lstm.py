import json

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, TABLES_DIR
from src.lstm_data import load_lstm_data
from src.lstm_model import LSTMClassifier


def evaluate_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return accuracy, macro_f1, all_labels, all_preds


def train_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, test_df, train_dataset, test_dataset, word_to_idx, label_to_idx, idx_to_label = load_lstm_data()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(
        vocab_size=len(word_to_idx),
        num_classes=len(label_to_idx)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        test_accuracy, test_macro_f1, _, _ = evaluate_model(model, test_loader, device)

        print(f"\nEpoch {epoch + 1} Summary")
        print("------------------------")
        print(f"Average Train Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Macro F1: {test_macro_f1:.4f}\n")

    final_accuracy, final_macro_f1, all_labels, all_preds = evaluate_model(model, test_loader, device)

    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "accuracy": final_accuracy,
        "macro_f1": final_macro_f1,
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df),
        "num_classes": len(label_to_idx),
        "vocab_size": len(word_to_idx),
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = TABLES_DIR / "lstm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    report_df = pd.DataFrame(report).transpose()
    report_path = TABLES_DIR / "lstm_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    model_path = TABLES_DIR / "lstm_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word_to_idx": word_to_idx,
            "label_to_idx": label_to_idx,
        },
        model_path,
    )

    print("Final LSTM Results")
    print("------------------")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Macro F1: {final_macro_f1:.4f}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved classification report to: {report_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_lstm()
