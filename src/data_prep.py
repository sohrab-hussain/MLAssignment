"""Step 1 - clean the raw UCI file and create a fixed train/test split.

Run once from the project root:
    python -m src.data_prep

Outputs
    data/train_data.csv  - used to fit every model
    test_data.csv        - held out; this is the file you upload in the Streamlit app
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import (
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)


def load_raw_data() -> pd.DataFrame:
    # The UCI file is semicolon separated, not comma separated.
    return pd.read_csv(RAW_DATA_PATH, sep=";")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    df = df.drop(columns=LEAKY_COLUMNS)

    # 'unknown' is kept as its own category: in this dataset it is informative
    # (e.g. clients with unknown credit default status behave differently).
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def main() -> None:
    df = clean(load_raw_data())
    print(f"Clean dataset: {df.shape[0]} rows, {df.shape[1] - 1} features")

    # stratify keeps the ~11% 'yes' rate identical in train and test
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    for name, part in [("train", train_df), ("test", test_df)]:
        rate = (part[TARGET_COLUMN] == "yes").mean()
        print(f"{name:5s}: {len(part):6d} rows, positive rate {rate:.2%}")


if __name__ == "__main__":
    main()
