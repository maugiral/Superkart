import os

import pandas as pd
import requests
import streamlit as st

# The Flask backend is reachable at this address inside the Docker network
# (service name "backend" is defined when the two containers are started together)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Prediction", layout="centered")
st.title("SuperKart Sales Prediction")
st.write(
    "Forecast the sales revenue of a product at a store, either one at a time "
    "(Online Inference) or for a whole file of records at once (Batch Inference)."
)

tab_online, tab_batch = st.tabs(["Online Inference", "Batch Inference"])

# ----------------------------- Online Inference -----------------------------
with tab_online:
    st.subheader("Single Prediction")

    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
        product_sugar_content = st.selectbox(
            "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
        )
        product_allocated_area = st.number_input(
            "Product Allocated Area", min_value=0.0, max_value=1.0, value=0.03
        )
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
        product_id_char = st.selectbox("Product Id Prefix", ["FD", "DR", "NC"])

    with col2:
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
        store_location_city_type = st.selectbox(
            "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
        )
        store_type = st.selectbox(
            "Store Type",
            ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
        )
        store_age_years = st.number_input("Store Age (Years)", min_value=0, value=16)
        product_type_category = st.selectbox(
            "Product Type Category", ["Perishables", "Non Perishables"]
        )

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age_years,
            "Product_Type_Category": product_type_category,
        }
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if response.status_code == 200:
                pred = response.json()["Product_Store_Sales_Total_Prediction"]
                st.success(f"Predicted Sales: {pred}")
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Could not reach the backend: {e}")

# ----------------------------- Batch Inference -----------------------------
with tab_batch:
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if st.button("Run Batch Prediction"):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files, timeout=60)
                if response.status_code == 200:
                    predictions = response.json()
                    batch_df["Predicted_Sales"] = [
                        predictions[str(i)] for i in range(len(batch_df))
                    ]
                    st.write("Predictions:")
                    st.dataframe(batch_df)
                    st.download_button(
                        "Download Predictions as CSV",
                        batch_df.to_csv(index=False),
                        "superkart_predictions.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Could not reach the backend: {e}")
