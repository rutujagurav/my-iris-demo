# train_export_models.py
"""
Train several classifiers on the iris dataset and export
their parameters to docs/models.json for client-side inference.
Models included:
 - logistic (multinomial)
 - random_forest
 - svm (probability=True)
 - mlp (sklearn MLPClassifier)

The frontend expects:
{
  "feature_names": [...],
  "class_names": [...],
  "scaler_mean": [...],
  "scaler_scale": [...],
  "models": {
     "logistic": { "type": "linear", "coef": [[...],...], "intercept": [...] },
     "random_forest": { "type": "rf", "trees": [ { "classes": [...], "feature_importances": [...], "thresholds": ... } , ... ] },
     "svm": { "type": "linear", "coef": [...], "intercept": [...] },
     "mlp": { "type": "mlp", "weights": [...], "biases": [...] }
  }
}
Notes: random_forest is exported as a simple probabilistic leaf-aggregation using sklearn.tree; MLP exported as dense layers for inference.
"""

import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.tree import _tree

# Load data
data = load_iris()
X = data['data']
y = data['target']
class_names = data['target_names'].tolist()
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Fit scaler
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Train models
log = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=400).fit(Xs, y)
rf  = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xs, y)
svm = SVC(kernel='linear', probability=True).fit(Xs, y)
mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1000, random_state=0).fit(Xs, y)

# Helper to export RF trees as simple structure
def export_tree(tree):
    # A compact representation: for each node we store left, right, feature, threshold, value (class counts)
    T = tree.tree_
    nodes = []
    for i in range(T.node_count):
        node = {
            "left": int(T.children_left[i]),
            "right": int(T.children_right[i]),
            "feature": int(T.feature[i]),
            "threshold": float(T.threshold[i]),
            # convert value shape (1, n_classes) to list
            "value": T.value[i][0].tolist()
        }
        nodes.append(node)
    return nodes

# Export RF as list of trees
rf_trees = []
for est in rf.estimators_:
    rf_trees.append(export_tree(est))

# Export MLP weights and biases
mlp_coefs = [w.tolist() for w in mlp.coefs_]   # list of arrays
mlp_intercepts = [b.tolist() for b in mlp.intercepts_]

models = {
    "logistic": {
        "type": "linear",
        "coef": log.coef_.tolist(),
        "intercept": log.intercept_.tolist()
    },
    "random_forest": {
        "type": "rf",
        "n_classes": int(rf.n_classes_),
        "trees": rf_trees
    },
    "svm": {
        "type": "linear",
        "coef": svm.coef_.tolist(),
        "intercept": svm.intercept_.tolist()
    },
    "mlp": {
        "type": "mlp",
        "coefs": mlp_coefs,
        "intercepts": mlp_intercepts,
        "activation": mlp.activation  # 'relu' or similar
    }
}

export = {
    "feature_names": feature_names,
    "class_names": class_names,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "models": models
}

# Save to docs/models.json (create docs/ if it doesn't exist)
import os
os.makedirs("docs", exist_ok=True)
with open("docs/models.json", "w") as f:
    json.dump(export, f, indent=2)

print("Wrote docs/models.json with models:", list(models.keys()))