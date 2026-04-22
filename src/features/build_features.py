import pandas as pd

# Known binary columns in Telco churn dataset
KNOWN_BINARY_COLUMNS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}


def _map_binary_series(s: pd.Series, col_name: str | None = None) -> pd.Series:
    """
    Apply deterministic binary encoding.

    Priority:
    1. Use fixed known mappings for known binary columns
    2. Use standard Yes/No mapping
    3. Use standard Male/Female mapping
    4. Use stable alphabetical mapping for any other 2-category feature
    """
    s = s.astype(str).str.strip()

    # 1. Known fixed mappings
    if col_name in KNOWN_BINARY_COLUMNS:
        mapping = KNOWN_BINARY_COLUMNS[col_name]
        return s.map(mapping).astype("Int64")

    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # 2. Standard Yes/No mapping
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")

    # 3. Standard Male/Female mapping
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # 4. Generic binary mapping
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.map(mapping).astype("Int64")

    # Non-binary feature
    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for both training and serving data.
    """
    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    # Identify object columns except target
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"   📊 Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # Always force known binary columns to binary if present
    known_binary_present = [c for c in KNOWN_BINARY_COLUMNS if c in obj_cols]

    # Infer other binary columns
    inferred_binary_cols = [
        c for c in obj_cols
        if df[c].dropna().nunique() == 2 and c not in known_binary_present
    ]

    binary_cols = known_binary_present + inferred_binary_cols
    multi_cols = [c for c in obj_cols if c not in binary_cols]

    print(f"   🔢 Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"      Binary: {binary_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")

    # Apply binary encoding
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c], col_name=c)
        print(f"      ✅ {c}: {original_dtype} → binary (0/1)")

    # Convert bool to int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   🔄 Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    # One-hot encode remaining multi-category columns
    if multi_cols:
        print(f"   🌟 Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      ✅ Created {new_features} new features from {len(multi_cols)} categorical columns")

    # Convert nullable ints to normal ints
    for c in binary_cols:
        if c in df.columns and pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    return df