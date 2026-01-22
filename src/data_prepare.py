import pandas as pd
import numpy as np
import torch

NUMERIC_COLS = [
    "VISIT",
    "THERCODE",
    "THERCOD1",
    "THERCOD2",
    "THERPHAS",
    "AGE",
]

CATEGORICAL_COLS = [
    "PROTOCOL",
    "THERAPY_STATUS",
    "ORIGIN",
    "GENDER",
    "GEOCODE",
]

def load_and_create_outcome(csv_path):
    df = pd.read_csv(csv_path)

    # HAMD columns
    hamd_cols = [f"HAMD{str(i).zfill(2)}" for i in range(1, 18)]

    # outcome y
    df["y"] = df[hamd_cols].sum(axis=1)

    return df


def patient_level_split(df, patient_col="UNIQUEID",
                        train_frac=0.7, val_frac=0.15, test_frac=0.15,
                        seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    patients = df[patient_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_total = len(patients)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    train_df = df[df[patient_col].isin(train_patients)].copy()
    val_df   = df[df[patient_col].isin(val_patients)].copy()
    test_df  = df[df[patient_col].isin(test_patients)].copy()

    return train_df, val_df, test_df

def list_covariate_columns(df):
    exclude_cols = (
        ["UNIQUEID", "THERAPY", "y"] +
        [f"HAMD{str(i).zfill(2)}" for i in range(1, 18)]
    )
    covariate_cols = [c for c in df.columns if c not in exclude_cols]
    return covariate_cols

def fit_numeric_standardizer(train_df, numeric_cols):
    stats = {}
    for col in numeric_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()
        if std == 0 or pd.isna(std):
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_numeric_standardizer(df, numeric_cols, stats):
    df = df.copy()
    for col in numeric_cols:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        df[col] = (df[col] - mean) / std
        df[col] = df[col].fillna(0.0)
    return df

def fit_categorical_encoders(train_df, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        values = train_df[col].astype(str).unique().tolist()
        mapping = {v: i for i, v in enumerate(values)}
        unknown_id = len(mapping)
        encoders[col] = {
            "mapping": mapping,
            "unknown": unknown_id,
        }
    return encoders


def apply_categorical_encoders(df, categorical_cols, encoders):
    df = df.copy()
    for col in categorical_cols:
        mapping = encoders[col]["mapping"]
        unknown = encoders[col]["unknown"]
        df[col] = (
            df[col]
            .astype(str)
            .map(mapping)
            .fillna(unknown)
            .astype(int)
        )
    return df

def get_category_sizes(encoders, categorical_cols):
    sizes = []
    for col in categorical_cols:
        # mapping size + unknown
        size = len(encoders[col]["mapping"]) + 1
        sizes.append(size)
    return sizes


def fit_treatment_mapping(train_df, treatment_col="THERAPY"):
    treatments = sorted(train_df[treatment_col].astype(str).unique())
    mapping = {t: i for i, t in enumerate(treatments)}
    return mapping


def apply_treatment_mapping(df, mapping, treatment_col="THERAPY"):
    df = df.copy()
    df["t_id"] = (
        df[treatment_col]
        .astype(str)
        .map(mapping)
        .astype(int)
    )
    return df

def build_model_inputs(df):
    x_num = torch.tensor(df[NUMERIC_COLS].values, dtype=torch.float32)
    x_cat = torch.tensor(df[CATEGORICAL_COLS].values, dtype=torch.long)
    t     = torch.tensor(df["t_id"].values, dtype=torch.long)
    y     = torch.tensor(df["y"].values, dtype=torch.float32)
    return x_num, x_cat, t, y

if __name__ == "__main__":

    # 1. Load data
    df = load_and_create_outcome("data/data_generated.csv")

    # 2. Split
    train_df, val_df, test_df = patient_level_split(df)

    # 3. Preprocess covariates
    num_stats = fit_numeric_standardizer(train_df, NUMERIC_COLS)
    cat_encoders = fit_categorical_encoders(train_df, CATEGORICAL_COLS)

    train_df = apply_numeric_standardizer(train_df, NUMERIC_COLS, num_stats)
    val_df   = apply_numeric_standardizer(val_df, NUMERIC_COLS, num_stats)
    test_df  = apply_numeric_standardizer(test_df, NUMERIC_COLS, num_stats)

    train_df = apply_categorical_encoders(train_df, CATEGORICAL_COLS, cat_encoders)
    val_df   = apply_categorical_encoders(val_df, CATEGORICAL_COLS, cat_encoders)
    test_df  = apply_categorical_encoders(test_df, CATEGORICAL_COLS, cat_encoders)

    # 4. Map treatments
    treatment_mapping = fit_treatment_mapping(train_df)
    train_df = apply_treatment_mapping(train_df, treatment_mapping)
    val_df   = apply_treatment_mapping(val_df, treatment_mapping)
    test_df  = apply_treatment_mapping(test_df, treatment_mapping)

    # 5. Build model inputs 
    x_num, x_cat, t, y = build_model_inputs(train_df)
    category_sizes = get_category_sizes(cat_encoders, CATEGORICAL_COLS)
    print("Category sizes:", category_sizes)


    print("x_num shape:", x_num.shape)
    print("x_cat shape:", x_cat.shape)
    print("t shape    :", t.shape)
    print("y shape    :", y.shape)

