import re
import pandas as pd

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataset(
    input_filename: str = "gretel_symptom_to_diagnosis.csv",
    output_filename: str = "processed_symptom_to_diagnosis.csv"
) -> pd.DataFrame:
    input_path = DATA_RAW_DIR / input_filename
    output_path = DATA_PROCESSED_DIR / output_filename

    df = pd.read_csv(input_path)

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved processed dataset to: {output_path}")
    return df


if __name__ == "__main__":
    df = preprocess_dataset()
    print("\nFirst 5 processed rows:\n")
    print(df.head())
    print("\nShape:\n")
    print(df.shape)
