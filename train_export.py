# train_export.py
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load iris
data = load_iris()
X = data['data']          # shape (150,4)
y = data['target']        # 0,1,2
class_names = data['target_names'].tolist()  # ['setosa', 'versicolor', 'virginica']

# Scale features
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Train multinomial logistic regression
clf = LogisticRegression(solver='lbfgs', max_iter=200)
clf.fit(Xs, y)

# Get parameters
params = {
    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coef": clf.coef_.tolist(),        # shape (3,4)
    "intercept": clf.intercept_.tolist(), # length 3
    "class_names": class_names
}

# Save JSON for frontend
with open("model_params.json", "w") as f:
    json.dump(params, f, indent=2)

print("Saved model_params.json")
print("Classes:", class_names)
print("Coefficients:")
print(np.array(params["coef"]))
print("Intercepts:", params["intercept"])
