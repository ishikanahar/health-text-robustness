import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DATA_PROCESSED_DIR, TABLES_DIR
from src.lstm_data import load_lstm_data, SymptomDataset
from src.lstm_model import LSTMClassifier
from src.perturb import synonym_replace, random_typo, drop_words


def make_perturbed_dataset(df, word_to_idx, label_to_idx, perturb_fn):
    perturbed_df = df.copy()
    perturbed_df["text"] = perturbed_df["text"].apply(perturb_fn)
    return SymptomDataset(perturbed_df, word_to_idx, label_to_idx)


def predict_lstm(model, dataloader, device):
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

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1


def evaluate_lstm_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, test_df, train_dataset, test_dataset, word_to_idx, label_to_idx, idx_to_label = load_lstm_data()

    checkpoint = torch.load(TABLES_DIR / "lstm_model.pt", map_location=device)

    model = LSTMClassifier(
        vocab_size=len(checkpoint["word_to_idx"]),
        num_classes=len(checkpoint["label_to_idx"])
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    scenarios = {
        "clean": test_df.copy(),
        "synonym": test_df.copy(),
        "typo": test_df.copy(),
        "drop": test_df.copy(),
    }

    scenarios["synonym"]["text"] = scenarios["synonym"]["text"].apply(synonym_replace)
    scenarios["typo"]["text"] = scenarios["typo"]["text"].apply(lambda x: random_typo(x, prob=0.1))
    scenarios["drop"]["text"] = scenarios["drop"]["text"].apply(lambda x: drop_words(x, prob=0.1))

    results = {}

    for scenario_name, scenario_df in scenarios.items():
        scenario_dataset = SymptomDataset(scenario_df, word_to_idx, label_to_idx)
        scenario_loader = DataLoader(scenario_dataset, batch_size=BATCH_SIZE, shuffle=False)

        acc, macro_f1 = predict_lstm(model, scenario_loader, device)
        results[scenario_name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
        }

    print("\nLSTM Robustness Results")
    print("-----------------------")
    for name, metrics in results.items():
        print(f"{name}: acc={metrics['accuracy']:.4f}, f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    evaluate_lstm_robustness()
