import pandas as pd
from datasets import load_dataset

from src.config import DATA_RAW_DIR


def load_gretel_dataset() -> pd.DataFrame:
    """
    Load the gretelai/symptom_to_diagnosis dataset from Hugging Face
    and convert it into a pandas DataFrame with columns:
    text, label, split
    """
    dataset = load_dataset("gretelai/symptom_to_diagnosis")

    train_df = pd.DataFrame(dataset["train"])
    train_df["split"] = "train"

    test_df = pd.DataFrame(dataset["test"])
    test_df["split"] = "test"

    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.rename(columns={"input_text": "text", "output_text": "label"})

    return df[["text", "label", "split"]]


def save_raw_dataset(
    df: pd.DataFrame,
    filename: str = "gretel_symptom_to_diagnosis.csv"
) -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved raw dataset to: {output_path}")


if __name__ == "__main__":
    df = load_gretel_dataset()
    print("\nFirst 5 rows:\n")
    print(df.head())
    print("\nTop label counts:\n")
    print(df["label"].value_counts().head(10))
    save_raw_dataset(df)
