
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_crop_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Fixed random seed for reproducibility
RANDOM_STATE = 42


def create_data_split():

    # Load cleaned dataset
    df = pd.read_csv(INPUT_FILE)

    # Separate features and target
    X = df.drop(columns=["low_yield_risk"])
    y = df["low_yield_risk"]

    # Perform stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Save training and testing data
    train_data = X_train.copy()
    train_data["low_yield_risk"] = y_train

    test_data = X_test.copy()
    test_data["low_yield_risk"] = y_test

    train_data.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_data.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("Train-Test Split Completed!")
    print("Training Samples:", len(train_data))
    print("Testing Samples:", len(test_data))
    print("Random State:", RANDOM_STATE)


if __name__ == "__main__":
    create_data_split()

