import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1500

brands = ["Maruti", "Hyundai", "Honda", "Toyota", "BMW"]
brand = np.random.choice(brands, rows)

mileage = np.random.randint(10, 30, rows)          # km/l
engine_size = np.random.randint(800, 3000, rows)   # cc

brand_price_map = {
    "Maruti": 300000,
    "Hyundai": 400000,
    "Honda": 500000,
    "Toyota": 600000,
    "BMW": 1500000
}

price = []
for b, m, e in zip(brand, mileage, engine_size):
    base = brand_price_map[b]
    car_price = base + (e * 150) - (m * 8000) + np.random.randint(-50000, 50000)
    price.append(car_price)

df = pd.DataFrame({
    "Brand": brand,
    "Mileage": mileage,
    "Engine_Size": engine_size,
    "Price": price
})

df.to_csv("car_data.csv", index=False)
print("✅ Dataset generated successfully (car_data.csv)")
