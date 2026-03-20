from pathlib import Path
import pandas as pd

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

summary_df = pd.DataFrame([
    {
        "Model": "TF-IDF",
        "Clean_Acc": 0.9700,
        "Synonym_Acc": 0.9500,
        "Typo_Acc": 0.7900,
        "Drop_Acc": 0.9400,
        "Clean_F1": 0.7566,
        "Synonym_F1": 0.6893,
        "Typo_F1": 0.4971,
        "Drop_F1": 0.6842,
    },
    {
        "Model": "LSTM",
        "Clean_Acc": 0.4100,
        "Synonym_Acc": 0.4000,
        "Typo_Acc": 0.2200,
        "Drop_Acc": 0.3600,
        "Clean_F1": 0.3407,
        "Synonym_F1": 0.3357,
        "Typo_F1": 0.1775,
        "Drop_F1": 0.3007,
    },
    {
        "Model": "Transformer",
        "Clean_Acc": 0.8700,
        "Synonym_Acc": 0.8600,
        "Typo_Acc": 0.4500,
        "Drop_Acc": 0.8500,
        "Clean_F1": 0.8575,
        "Synonym_F1": 0.8468,
        "Typo_F1": 0.4351,
        "Drop_F1": 0.8293,
    }
])

# add drop-from-clean columns
summary_df["Synonym_Acc_Drop"] = summary_df["Clean_Acc"] - summary_df["Synonym_Acc"]
summary_df["Typo_Acc_Drop"] = summary_df["Clean_Acc"] - summary_df["Typo_Acc"]
summary_df["Dropword_Acc_Drop"] = summary_df["Clean_Acc"] - summary_df["Drop_Acc"]

output_path = results_dir / "final_summary_table.csv"
summary_df.to_csv(output_path, index=False)

print("Saved final summary table to:")
print(output_path.resolve())
print("\nPreview:")
print(summary_df.to_string(index=False))
