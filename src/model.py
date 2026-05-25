import pandas as pd
import numpy as np
import joblib  # <-- ADD THIS IMPORT
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.data_processing import get_data

def train_and_evaluate():
    print("--- Step 2 & 3: Training and Evaluating Model ---")
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {X_train.shape[0]} training rows, {X_test.shape[0]} testing rows.")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    print("\n--- Model Coefficients ---")
    print(coefficients.to_string(index=False))
    
    # <-- ADD THESE TWO LINES TO SAVE THE MODEL -->
    joblib.dump(model, 'models/house_price_model.pkl')
    print("\nModel successfully saved to 'models/house_price_model.pkl'")
    # -------------------------------------------
    
    return model

if __name__ == "__main__":
    train_and_evaluate()