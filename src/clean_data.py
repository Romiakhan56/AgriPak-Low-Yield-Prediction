
import pandas as pd
from pathlib import Path

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "crop_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_crop_data.csv"


def load_and_clean_data():

    # Load original dataset
    df = pd.read_csv(RAW_FILE)

    print("Original Dataset Shape:", df.shape)

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Remove extra spaces from categorical values
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].str.strip()

    # Convert numerical columns to appropriate types
    numerical_cols = [
        "rainfall_mm",
        "temperature_c",
        "fertilizer_used",
        "irrigation_used",
        "days_to_harvest",
        "low_yield_risk"
    ]

    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicate records
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    # Handle missing values by removing incomplete rows
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_after = df.isnull().sum().sum()

    # Convert binary columns to integers
    binary_cols = [
        "fertilizer_used",
        "irrigation_used",
        "low_yield_risk"
    ]

    for col in binary_cols:
        df[col] = df[col].astype(int)

    # Save cleaned dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nData Cleaning Completed!")
    print("Duplicate Rows Removed:", duplicates)
    print("Missing Values Before:", missing_before)
    print("Missing Values After:", missing_after)
    print("Final Dataset Shape:", df.shape)
    print("Cleaned File Saved At:", OUTPUT_FILE)


if __name__ == "__main__":
    load_and_clean_data()

