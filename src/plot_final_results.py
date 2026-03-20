from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

models = ["TF-IDF", "LSTM", "Transformer"]

clean_acc = [0.97, 0.41, 0.87]
syn_acc = [0.95, 0.40, 0.86]
typo_acc = [0.79, 0.22, 0.45]
drop_acc = [0.94, 0.36, 0.85]

clean_f1 = [0.7566, 0.3407, 0.8575]
syn_f1 = [0.6893, 0.3357, 0.8468]
typo_f1 = [0.4971, 0.1775, 0.4351]
drop_f1 = [0.6842, 0.3007, 0.8293]

x = np.arange(len(models))
width = 0.2

# Accuracy plot
plt.figure(figsize=(10, 6))
plt.bar(x - 1.5 * width, clean_acc, width, label="Clean")
plt.bar(x - 0.5 * width, syn_acc, width, label="Synonym")
plt.bar(x + 0.5 * width, typo_acc, width, label="Typo")
plt.bar(x + 1.5 * width, drop_acc, width, label="Drop")

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.title("Accuracy Across Perturbations")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "accuracy_across_perturbations.png", dpi=300)
plt.close()

# F1 plot
plt.figure(figsize=(10, 6))
plt.bar(x - 1.5 * width, clean_f1, width, label="Clean")
plt.bar(x - 0.5 * width, syn_f1, width, label="Synonym")
plt.bar(x + 0.5 * width, typo_f1, width, label="Typo")
plt.bar(x + 1.5 * width, drop_f1, width, label="Drop")

plt.xticks(x, models)
plt.ylabel("Macro F1")
plt.ylim(0, 1.0)
plt.title("F1 Score Across Perturbations")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "f1_across_perturbations.png", dpi=300)
plt.close()

# Save summary CSV
df = pd.DataFrame({
    "Model": models,
    "Clean_Acc": clean_acc,
    "Synonym_Acc": syn_acc,
    "Typo_Acc": typo_acc,
    "Drop_Acc": drop_acc,
    "Clean_F1": clean_f1,
    "Synonym_F1": syn_f1,
    "Typo_F1": typo_f1,
    "Drop_F1": drop_f1
})

df.to_csv(results_dir / "final_summary_table.csv", index=False)

print("Saved plots and summary table to results/")
