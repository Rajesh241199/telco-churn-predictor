from typing import Tuple, List
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Lightweight validation for Telco Customer Churn dataset.
    Returns:
        (is_valid, failed_checks)
    """
    failed = []

    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for col in required_columns:
        if col not in df.columns:
            failed.append(f"missing_column:{col}")

    if failed:
        return False, failed

    # customerID
    if df["customerID"].isna().any():
        failed.append("customerID_nulls")

    # allowed categorical values
    if not df["gender"].dropna().isin(["Male", "Female"]).all():
        failed.append("gender_invalid_values")

    for col in ["Partner", "Dependents", "PhoneService"]:
        if not df[col].dropna().isin(["Yes", "No"]).all():
            failed.append(f"{col}_invalid_values")

    if not df["Contract"].dropna().isin(["Month-to-month", "One year", "Two year"]).all():
        failed.append("Contract_invalid_values")

    if not df["InternetService"].dropna().isin(["DSL", "Fiber optic", "No"]).all():
        failed.append("InternetService_invalid_values")

    # numeric checks
    tenure_num = pd.to_numeric(df["tenure"], errors="coerce")
    monthly_num = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    total_num = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if tenure_num.isna().any():
        failed.append("tenure_non_numeric")
    if monthly_num.isna().any():
        failed.append("MonthlyCharges_non_numeric")

    # TotalCharges may have blanks in raw data, so allow coercion but check negative values only on valid rows
    if (tenure_num.dropna() < 0).any():
        failed.append("tenure_negative")
    if (tenure_num.dropna() > 120).any():
        failed.append("tenure_out_of_range")

    if (monthly_num.dropna() < 0).any():
        failed.append("MonthlyCharges_negative")
    if (monthly_num.dropna() > 200).any():
        failed.append("MonthlyCharges_out_of_range")

    if (total_num.dropna() < 0).any():
        failed.append("TotalCharges_negative")

    # soft consistency check: total charges usually >= monthly charges
    valid_pair = total_num.notna() & monthly_num.notna()
    if valid_pair.any():
        consistency_rate = (total_num[valid_pair] >= monthly_num[valid_pair]).mean()
        if consistency_rate < 0.95:
            failed.append("TotalCharges_vs_MonthlyCharges_inconsistent")

    return len(failed) == 0, failed