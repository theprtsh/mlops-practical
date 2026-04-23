import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import os

def train_models():
    # Load dataset
    if not os.path.exists('dataset/iris.csv'):
        print("Data not found. Run data.py first.")
        return

    df = pd.read_csv('dataset/iris.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Version 1: max_depth=2
    v1_depth = 2
    model_v1 = RandomForestClassifier(max_depth=v1_depth, random_state=42)
    model_v1.fit(X_train, y_train)
    score_v1 = model_v1.score(X_test, y_test)

    # Version 2: max_depth=5
    v2_depth = 5
    model_v2 = RandomForestClassifier(max_depth=v2_depth, random_state=42)
    model_v2.fit(X_train, y_train)
    score_v2 = model_v2.score(X_test, y_test)

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Save models
    joblib.dump(model_v1, 'models/model_v1.pkl')
    joblib.dump(model_v2, 'models/model_v2.pkl')

    # Save metrics
    metrics = {
        "v1": {"max_depth": v1_depth, "accuracy": score_v1},
        "v2": {"max_depth": v2_depth, "accuracy": score_v2}
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Models saved. V1 Accuracy: {score_v1:.2f}, V2 Accuracy: {score_v2:.2f}")

if __name__ == "__main__":
    train_models()
