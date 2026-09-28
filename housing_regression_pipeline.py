import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ============================================================
# 1. GENERATE SYNTHETIC HOUSING DATA
# ============================================================

np.random.seed(10)
n_houses = 1200

data = {
    'Square_Feet': np.random.normal(1800, 500, n_houses).astype(int),
    'Bedrooms': np.random.randint(1, 5, n_houses),
    'Bathrooms': np.random.randint(1, 4, n_houses),
    'Age_of_Property': np.random.randint(0, 40, n_houses),
    'Location_Tier': np.random.choice(
        [1, 2, 3],
        n_houses,
        p=[0.4, 0.4, 0.2]
    )
}

df_house = pd.DataFrame(data)

# ============================================================
# 2. CREATE REALISTIC PRICE TARGET
# ============================================================

location_effect = df_house['Location_Tier'].map({
    1: 500000,
    2: 250000,
    3: 0
})

df_house['Price'] = (
    df_house['Square_Feet'] * 150
    + df_house['Bedrooms'] * 10000
    + df_house['Bathrooms'] * 15000
    - df_house['Age_of_Property'] * 800
    + location_effect
    + np.random.normal(0, 15000, n_houses)
)

# ============================================================
# 3. OUTLIER TREATMENT
# ============================================================

q_low = df_house['Square_Feet'].quantile(0.01)
q_hi = df_house['Square_Feet'].quantile(0.99)

df_house = df_house[
    (df_house['Square_Feet'] < q_hi)
    & (df_house['Square_Feet'] > q_low)
]

# ============================================================
# 4. ENCODE CATEGORICAL FEATURE
# ============================================================

df_house = pd.get_dummies(
    df_house,
    columns=['Location_Tier'],
    drop_first=True
)

# Separate features and target
X = df_house.drop(columns=['Price'])
y = df_house['Price']

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 7. GRADIENT BOOSTING REGRESSOR
# ============================================================

gbr = GradientBoostingRegressor(
    n_estimators=120,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

gbr.fit(X_train_scaled, y_train)

# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = gbr.predict(X_test_scaled)

# ============================================================
# 9. EVALUATION
# ============================================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=" * 60)
print("HOUSING PRICE REGRESSION")
print("=" * 60)

print(f"\nR-squared Score: {r2:.3f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")

print("=" * 60)