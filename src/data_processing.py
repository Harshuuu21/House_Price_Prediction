import pandas as pd
from sklearn.datasets import fetch_california_housing

def get_data():
    """
    Task 1: Load the housing dataset and clean/explore features.
    (We will do the visual exploration in the notebook later, 
    but here we handle the programmatic loading and cleaning).
    """
    print("Loading California Housing dataset...")
    
    # 1. Load the dataset
    housing = fetch_california_housing()
    
    # 2. Convert to Pandas DataFrames/Series
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='Price')
    
    # 3. Clean: Check for missing values (Drop them if any exist)
    # The sklearn dataset is perfectly clean, but this is best practice
    initial_rows = X.shape[0]
    X = X.dropna()
    y = y[X.index] # Ensure target 'y' matches the dropped rows in 'X'
    
    if initial_rows != X.shape[0]:
        print(f"Cleaned data: Dropped {initial_rows - X.shape[0]} rows with missing values.")
    else:
        print("Data is already clean. No missing values found.")
        
    print(f"Final dataset shape: {X.shape[0]} rows, {X.shape[1]} features.\n")
    
    return X, y

# This allows us to test the file directly if we run it by itself
if __name__ == "__main__":
    X, y = get_data()
    print(X.head())