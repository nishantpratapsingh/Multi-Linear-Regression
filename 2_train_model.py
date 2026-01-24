import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset
data = pd.read_csv("car_data.csv")

X = data[["Brand", "Mileage", "Engine_Size"]]
y = data["Price"]

# Preprocessing: One-Hot Encoding for Brand
preprocessor = ColumnTransformer(
    transformers=[
        ("brand", OneHotEncoder(drop="first"), ["Brand"])
    ],
    remainder="passthrough"
)

X_processed = preprocessor.fit_transform(X)

# Train Multiple Linear Regression model
model = LinearRegression()
model.fit(X_processed, y)

# Feature importance
feature_names = (
    preprocessor.named_transformers_["brand"]
    .get_feature_names_out(["Brand"])
    .tolist()
    + ["Mileage", "Engine_Size"]
)

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_
})

print("\n📊 Feature Importance (Model Coefficients):")
print(coefficients)

# Save model and preprocessor
with open("model.pkl", "wb") as file:
    pickle.dump((model, preprocessor), file)

print("\n✅ Model and preprocessor saved successfully")
