import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load Data using the new caching method
@st.cache_data
def load_data():
    data = pd.read_csv('fixed_assets.csv')  # Replace with your file path
    return data

data = load_data()

# Display Data
st.header("Oracle ERP Fixed Asset Details")
st.write("--------------------------------------", data)

# Prepare data for model training
X = data.drop(columns=['Retired', 'Asset_ID', 'Acquisition_Date'])
y = data['Retired']

# One-Hot Encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
#st.write("Accuracy:", accuracy_score(y_test, y_pred))
#st.write("Classification Report:", classification_report(y_test, y_pred))

# User Input for a New Asset
st.header("Enter Asset Details Retirement Readiness Analysis.")

# Input fields for asset features
asset_type = st.selectbox("Asset Type", options=['Machine', 'Vehicle', 'Building', 'Furniture', 'Equipment'])
acquisition_cost = st.number_input("Acquisition Cost", min_value=5000, max_value=250000, step=1000)
acquisition_date = st.date_input("Acquisition Date", value=pd.to_datetime("2000-01-01"))
depreciation_method = st.selectbox("Depreciation Method", options=['Straight-Line', 'Declining Balance'])
depreciation_rate = st.number_input("Depreciation Rate (%)", min_value=5, max_value=20, step=1)
useful_life = st.number_input("Useful Life (years)", min_value=5, max_value=30, step=1)
maintenance_cost = st.number_input("Total Maintenance Cost", min_value=0, step=100)
repair_frequency = st.number_input("Repair Frequency", min_value=0, step=1)
condition_score = st.number_input("Condition Score (1-10)", min_value=1, max_value=10, step=1)
utilization_percentage = st.number_input("Utilization Percentage (%)", min_value=0, max_value=100, step=1)

if st.button('Predict Retirement Readiness'):
    # Prepare the input data for prediction
    new_asset = pd.DataFrame({
        'Asset_Type': [asset_type],
        'Acquisition_Cost': [acquisition_cost],
        'Depreciation_Method': [depreciation_method],
        'Depreciation_Rate': [depreciation_rate],
        'Useful_Life': [useful_life],
        'Maintenance_Cost': [maintenance_cost],
        'Repair_Frequency': [repair_frequency],
        'Condition_Score': [condition_score],
        'Utilization_Percentage': [utilization_percentage]
    })

    # One-Hot Encoding for the new asset
    new_asset_encoded = pd.get_dummies(new_asset, drop_first=True)

    # Align the columns
    new_asset_encoded = new_asset_encoded.reindex(columns=X.columns, fill_value=0)

    # Make Prediction
    prediction = model.predict(new_asset_encoded)
    st.write("Retirement Status (1=Ready to Retire , 0= Not Ready to be Retired):", prediction[0])


