import os
import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://3.26.209.36:8000/predict")


def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        churn_prob = result.get("churn_probability", None)
        churn_pred = result.get("churn_prediction", None)
        threshold = result.get("threshold", None)

        if churn_prob is not None:
            if churn_prob >= 0.70:
                risk = "High Risk"
            elif churn_prob >= 0.35:
                risk = "Medium Risk"
            else:
                risk = "Low Risk"
        else:
            risk = "Unknown"

        output = (
            f"Prediction: {'Likely to Churn' if churn_pred == 1 else 'Not Likely to Churn'}\n"
            f"Churn Probability: {churn_prob:.4f}\n"
            f"Threshold: {threshold}\n"
            f"Risk Level: {risk}"
        )
        return output

    except requests.exceptions.RequestException as e:
        return f"API request failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


custom_css = """
body, .gradio-container {
    background: #050b1a !important;
    color: white !important;
    font-family: Arial, sans-serif;
}
.block, .gr-box, .gr-panel {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
}
h1, h2, h3, label, .gr-form, .gr-textbox, .gr-dropdown {
    color: white !important;
}
textarea, input, select {
    background: #111827 !important;
    color: white !important;
    border: 1px solid #374151 !important;
}
button {
    background: #4b5563 !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
}
button:hover {
    background: #6b7280 !important;
}
"""


with gr.Blocks(css=custom_css, title="Telco Churn Predictor") as demo:
    gr.Markdown(
        """
        # Telco Churn Predictor
        Fill in the customer details to get a churn prediction.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
            senior_citizen = gr.Dropdown([0, 1], label="Senior Citizen", value=0)
            partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
            tenure = gr.Number(label="Tenure", value=5)
            phone_service = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
            multiple_lines = gr.Dropdown(
                ["No", "Yes", "No phone service"],
                label="Multiple Lines",
                value="No",
            )
            internet_service = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet Service",
                value="Fiber optic",
            )
            online_security = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Security",
                value="No",
            )
            online_backup = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Backup",
                value="No",
            )
            device_protection = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Device Protection",
                value="No",
            )
            tech_support = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Tech Support",
                value="No",
            )
            streaming_tv = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming TV",
                value="Yes",
            )
            streaming_movies = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming Movies",
                value="Yes",
            )
            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract",
                value="Month-to-month",
            )
            paperless_billing = gr.Dropdown(
                ["Yes", "No"],
                label="Paperless Billing",
                value="Yes",
            )
            payment_method = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                label="Payment Method",
                value="Electronic check",
            )
            monthly_charges = gr.Number(label="Monthly Charges", value=70.35)
            total_charges = gr.Number(label="Total Charges", value=350.75)

        with gr.Column(scale=2):
            output = gr.Textbox(label="Output", lines=8)
            predict_btn = gr.Button("Flag")

    predict_btn.click(
        fn=predict_churn,
        inputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)