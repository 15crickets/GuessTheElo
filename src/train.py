import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 


from xgboost import XGBRegressor


def convert_result(row):
    if row["result"] == "Draw":
        return 0.5

    if row["result"] == "White":
        return 1.0 if row["color"] == "white" else 0.0

    if row["result"] == "Black":
        return 1.0 if row["color"] == "black" else 0.0

    

df = pd.read_csv("../data/features.csv")



print(f"Dataset shape: {df.shape}")
print(df["result"].unique())
print(df["color"].unique())
y = df["elo"]

df["result"] = df.apply(convert_result, axis=1)

drop_columns = [
    "elo",
    "termination",
    "result",
    "color"
]

X = df.drop(columns=drop_columns)

numeric = df.select_dtypes(include="number")

print("Elo correlations")
print(
    numeric.corr()["elo"]
    .sort_values()
    .to_string()
)



X["had_blunder"] = X["first_blunder_move"].notna().astype(int)

X["first_blunder_move"] = X["first_blunder_move"].fillna(-1)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

baseline_prediction = y_train.mean()
baseline_predictions = [baseline_prediction] * len(y_test)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)


print("\nBaseline Results")
print("=" * 40)
print(f"Baseline MAE: {baseline_mae:.2f} Elo")

model = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\nXGBoost Results")
print("=" * 40)

print(f"MAE:  {mae:.2f} Elo")
print(f"RMSE: {rmse:.2f} Elo")
print(f"R²:   {r2:.4f}")

model.save_model("../models/xgboost_model.json")