import io

import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Loading the serialized model (must already be in this folder before deployment)
MODEL_PATH = "superkart_model.joblib"
model = joblib.load(MODEL_PATH)

# The feature columns the model expects (order does not matter since the
# pipeline's ColumnTransformer selects columns by name)
FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]

superkart_api = Flask("SuperKart Sales Prediction")


@superkart_api.get("/")
def home():
    return {"message": "SuperKart sales prediction API is up and running."}


@superkart_api.post("/v1/predict")
def predict_single():
    """Online inference: predicts sales for a single product/store record sent as JSON."""
    payload = request.get_json()

    missing = [col for col in FEATURE_COLUMNS if col not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    input_df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]

    return jsonify({"Product_Store_Sales_Total_Prediction": round(float(prediction), 2)})


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    """Batch inference: predicts sales for every row in an uploaded CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' in the request."}), 400

    file = request.files["file"]
    input_df = pd.read_csv(io.BytesIO(file.read()))

    missing = [col for col in FEATURE_COLUMNS if col not in input_df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    predictions = model.predict(input_df[FEATURE_COLUMNS])
    result = {str(idx): round(float(pred), 2) for idx, pred in enumerate(predictions)}

    return jsonify(result)


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
