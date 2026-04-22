from pathlib import Path
from contextlib import asynccontextmanager
import sys
import json
import joblib
import pandas as pd
import mlflow.xgboost as mlflow_xgb

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# Project path setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# --------------------------------------------------
# Fixed deployment config from selected best run
# --------------------------------------------------
THRESHOLD = 0.35

EXPERIMENT_ID = "867506720463210619"
RUN_ID = "9b147a30bc2e442899b9be21679ee2c9"
MODEL_ID = "m-bcc65e5cb57c4ec5a6c84aa0858c25c0"

# Docker container path
MODEL_URI = (
    "file:///app/mlruns/867506720463210619/"
    "models/m-bcc65e5cb57c4ec5a6c84aa0858c25c0/"
    "artifacts"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
PREPROCESSING_PATH = ARTIFACTS_DIR / "preprocessing.pkl"

model = None
feature_columns = None
preprocessing_artifact = None


# --------------------------------------------------
# Input schema
# --------------------------------------------------
class CustomerInput(BaseModel):
    gender: str = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[5])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["Yes"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., examples=[70.35])
    TotalCharges: float = Field(..., examples=[350.75])


# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
def load_feature_artifacts():
    global feature_columns, preprocessing_artifact

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Missing feature columns file: {FEATURE_COLUMNS_PATH}")

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    if PREPROCESSING_PATH.exists():
        preprocessing_artifact = joblib.load(PREPROCESSING_PATH)
    else:
        preprocessing_artifact = None


def load_model():
    global model
    model = mlflow_xgb.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_feature_artifacts()
    load_model()
    yield


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    description="Predict churn probability and churn label for a telecom customer",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Telco Churn Prediction API is running",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "model_id": MODEL_ID,
        "threshold": THRESHOLD,
        "model_uri": MODEL_URI,
        "model_loaded": model is not None,
        "feature_columns_loaded": feature_columns is not None,
        "feature_count": len(feature_columns) if feature_columns else 0,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "model_id": MODEL_ID,
        "model_loaded": model is not None,
        "feature_columns_loaded": feature_columns is not None,
    }


@app.post("/predict")
def predict(payload: CustomerInput):
    try:
        # 1. Convert request payload to DataFrame
        df = pd.DataFrame([payload.model_dump()])

        # 2. Apply same preprocessing as training
        df = preprocess_data(df, target_col="Churn")

        # 3. Apply same feature engineering as training
        df_features = build_features(df, target_col="Churn")

        # 4. Align with exact training feature columns
        X = df_features.reindex(columns=feature_columns, fill_value=0)

        # 5. Predict probability
        churn_probability = float(model.predict_proba(X)[:, 1][0])

        # 6. Apply selected deployment threshold
        churn_prediction = int(churn_probability >= THRESHOLD)

        return {
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "model_id": MODEL_ID,
            "threshold": THRESHOLD,
            "churn_probability": round(churn_probability, 6),
            "churn_prediction": churn_prediction,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))