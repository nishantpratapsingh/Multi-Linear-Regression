import pickle
import pandas as pd

with open("model.pkl", "rb") as file:
    model, preprocessor = pickle.load(file)

sample_input = pd.DataFrame({
    "Brand": ["Honda"],
    "Mileage": [18],
    "Engine_Size": [1500]
})

processed_input = preprocessor.transform(sample_input)
prediction = model.predict(processed_input)

print(f"🚗 Predicted Car Price: ₹{int(prediction[0]):,}")
