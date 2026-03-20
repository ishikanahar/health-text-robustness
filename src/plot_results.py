import matplotlib.pyplot as plt
import numpy as np

# Your results (hardcoded for now)
models = ["TF-IDF", "LSTM", "Transformer"]

clean_acc = [0.97, 0.41, 0.87]
syn_acc = [0.95, 0.40, 0.86]
typo_acc = [0.79, 0.22, 0.45]
drop_acc = [0.94, 0.36, 0.85]

x = np.arange(len(models))
width = 0.2

plt.figure()

plt.bar(x - 1.5*width, clean_acc, width, label="Clean")
plt.bar(x - 0.5*width, syn_acc, width, label="Synonym")
plt.bar(x + 0.5*width, typo_acc, width, label="Typo")
plt.bar(x + 1.5*width, drop_acc, width, label="Drop")

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Model Performance Across Perturbations")
plt.legend()

plt.tight_layout()
plt.savefig("results/performance_comparison.png")
plt.show()
