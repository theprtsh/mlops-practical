import pandas as pd
from sklearn.datasets import load_iris
import os

def load_and_save_data():
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Create dataset directory
    os.makedirs('dataset', exist_ok=True)
    
    # Save to CSV
    df.to_csv('dataset/iris.csv', index=False)
    print("Dataset saved to dataset/iris.csv")

if __name__ == "__main__":
    load_and_save_data()
