from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# ── Train the model once at startup ──────────────────────────
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)
n = 2000

bedrooms    = np.random.randint(1, 7, n)
bathrooms   = np.random.randint(1, 5, n)
area_sqft   = np.random.randint(400, 6000, n)
age_years   = np.random.randint(0, 55, n)
garage      = np.random.randint(0, 3, n)
distance_km = np.round(np.random.uniform(1, 50, n), 1)
location    = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])
floor       = np.random.randint(1, 20, n)
furnished   = np.random.choice([0, 1], n)

price = (
    80000
    + bedrooms    * 28000
    + bathrooms   * 18000
    + area_sqft   * 155
    - age_years   * 2200
    + garage      * 12000
    - distance_km * 3200
    + location    * 45000
    + floor       * 1500
    + furnished   * 20000
    + np.random.normal(0, 18000, n)
)
price = np.maximum(price, 50000).astype(int)

FEATURES = ['bedrooms','bathrooms','area_sqft','age_years',
            'garage','distance_km','location','floor','furnished']

X = np.column_stack([bedrooms, bathrooms, area_sqft, age_years,
                     garage, distance_km, location, floor, furnished])
y = price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    "Ridge Regression":  Ridge(alpha=10),
}

stats = {}
trained = {}

for name, m in models.items():
    if name == "Ridge Regression":
        m.fit(X_train_sc, y_train)
        preds = m.predict(X_test_sc)
    else:
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

    r2   = round(r2_score(y_test, preds), 4)
    rmse = int(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = int(mean_absolute_error(y_test, preds))
    trained[name] = m
    stats[name]   = {"r2": r2, "rmse": rmse, "mae": mae,
                     "accuracy": round(r2 * 100, 1)}

best_model_name = max(stats, key=lambda k: stats[k]["r2"])
best_model      = trained[best_model_name]

print(f"✅ Models trained. Best: {best_model_name} R²={stats[best_model_name]['r2']}")


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        feat = np.array([[
            float(data["bedrooms"]),
            float(data["bathrooms"]),
            float(data["area_sqft"]),
            float(data["age_years"]),
            float(data["garage"]),
            float(data["distance_km"]),
            float(data["location"]),
            float(data["floor"]),
            float(data["furnished"]),
        ]])

        if best_model_name == "Ridge Regression":
            feat_sc = scaler.transform(feat)
            price   = int(best_model.predict(feat_sc)[0])
        else:
            price = int(best_model.predict(feat)[0])

        price = max(price, 50000)
        lo    = int(price * 0.88)
        hi    = int(price * 1.12)

        # per-model predictions
        all_preds = {}
        for name, m in trained.items():
            if name == "Ridge Regression":
                p = int(m.predict(scaler.transform(feat))[0])
            else:
                p = int(m.predict(feat)[0])
            all_preds[name] = max(p, 50000)

        return jsonify({
            "price": price,
            "low":   lo,
            "high":  hi,
            "model": best_model_name,
            "all_predictions": all_preds,
            "confidence": min(97, max(62, round(
                90 - abs(float(data["area_sqft"]) - 2000) / 120
                   - float(data["age_years"]) * 0.25
                   - float(data["distance_km"]) * 0.3, 1
            )))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stats")
def model_stats():
    return jsonify({
        "models": stats,
        "best":   best_model_name,
        "dataset_size": n,
        "features": FEATURES
    })


@app.route("/api/feature_importance")
def feature_importance():
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_.tolist()
    else:
        imp = (np.abs(best_model.coef_) / np.abs(best_model.coef_).sum()).tolist()
    return jsonify({"features": FEATURES, "importance": imp})


if __name__ == "__main__":
    print("\n🚀 Starting server at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
