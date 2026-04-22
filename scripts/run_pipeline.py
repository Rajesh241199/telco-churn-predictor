#!/usr/bin/env python3
"""
Runs sequentially:
load -> validate -> preprocess -> feature engineering -> train -> evaluate -> log to MLflow
"""

import os
import sys
import time
import json
import joblib
import argparse
from pathlib import Path

import mlflow
import mlflow.xgboost as mlflow_xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Make src/ importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def main(args):
    project_root = Path(__file__).resolve().parent.parent

    # MLflow setup
    mlruns_path = args.mlflow_uri or (project_root / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        # Log run config
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("target", args.target)

        # Stage 1: Load data
        print("🔄 Loading data...")
        df = load_data(args.input)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Stage 2: Validate raw data
        print("🔍 Validating data quality...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_checks.json")
            raise ValueError(f"Data quality check failed: {failed}")

        print("✅ Data validation passed")

        # Stage 3: Preprocess
        print("🔧 Preprocessing data...")
        df = preprocess_data(df, target_col=args.target)

        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found after preprocessing")

        processed_path = project_root / "data" / "processed" / "telco_churn_processed.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        mlflow.log_artifact(str(processed_path))
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # Stage 4: Feature engineering
        print("🛠️ Building features...")
        df_enc = build_features(df, target_col=args.target)

        bool_cols = df_enc.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            df_enc[bool_cols] = df_enc[bool_cols].astype(int)

        print(f"✅ Feature engineering completed: {df_enc.shape[1]} columns")

        # Save feature metadata
        artifacts_dir = project_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[args.target]).columns)

        feature_cols_path = artifacts_dir / "feature_columns.json"
        with open(feature_cols_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)

        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": args.target,
        }

        preprocessing_path = artifacts_dir / "preprocessing.pkl"
        joblib.dump(preprocessing_artifact, preprocessing_path)

        mlflow.log_artifact(str(feature_cols_path))
        mlflow.log_artifact(str(preprocessing_path))
        print(f"✅ Saved {len(feature_cols)} feature columns for serving consistency")

        # Stage 5: Train/test split
        print("📊 Splitting data...")
        X = df_enc.drop(columns=[args.target])
        y = df_enc[args.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=42,
        )

        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        mlflow.log_param("scale_pos_weight", float(scale_pos_weight))
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f}")

        # Stage 6: Train model
        print("🤖 Training XGBoost model...")

        model = XGBClassifier(
            n_estimators=651,
            learning_rate=0.028022265133981553,
            max_depth=9,
            subsample=0.952552437305337,
            colsample_bytree=0.8888332110085739,
            min_child_weight=8,
            gamma=3.7600351068474787,
            reg_alpha=3.141043534959303,
            reg_lambda=3.5740591519288993,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )

        mlflow.log_params(
            {
                "n_estimators": 651,
                "learning_rate": 0.028022265133981553,
                "max_depth": 9,
                "subsample": 0.952552437305337,
                "colsample_bytree": 0.8888332110085739,
                "min_child_weight": 8,
                "gamma": 3.7600351068474787,
                "reg_alpha": 3.141043534959303,
                "reg_lambda": 3.5740591519288993,
                "random_state": 42,
            }
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # Stage 7: Evaluate
        print("📊 Evaluating model...")

        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        report = classification_report(y_test, y_pred, digits=3)
        mlflow.log_text(report, artifact_file="classification_report.txt")

        print("\n🎯 Model Performance:")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall   : {recall:.3f}")
        print(f"   F1 Score : {f1:.3f}")
        print(f"   ROC AUC  : {roc_auc:.3f}")

        # Stage 8: Log model
        print("\n💾 Saving model to MLflow...")
        mlflow_xgb.log_model(model, artifact_path="model")
        print("✅ Model saved to MLflow")

        print("\n⏱️ Performance Summary:")
        print(f"   Training time : {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples/sec   : {len(X_test) / pred_time:.0f}")

        print("\n📈 Detailed Classification Report:")
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file (e.g., data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)",
    )
    parser.add_argument("--target", type=str, default="Churn")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--experiment", type=str, default="Telco Churn")
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="Optional MLflow tracking URI. Example: file:///C:/Users/RAJESH/Documents/Telco_churn/mlruns",
    )

    args = parser.parse_args()
    main(args)