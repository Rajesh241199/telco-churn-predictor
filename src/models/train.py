import mlflow
import pandas as pd
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def train_model(df: pd.DataFrame, target_col: str, threshold: float = 0.4):
    """
    Trains an XGBoost model with tuned parameters and logs to MLflow.

    Args:
        df (pd.DataFrame): Feature dataset
        target_col (str): Target column name
        threshold (float): Classification threshold for converting probabilities to labels
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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

    with mlflow.start_run():
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, proba)

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
                "scale_pos_weight": float(scale_pos_weight),
                "threshold": threshold,
                "test_size": 0.2,
                "random_state": 42,
            }
        )

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.xgboost.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")

    return model