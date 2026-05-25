import pandas as pd  # Change numpy to pandas
import joblib

def make_sample_predictions():
    print("\n--- Step 4: Loading Model and Making Predictions ---")
    model = joblib.load('models/house_price_model.pkl')
    print("Model loaded successfully!")
    
    # Change this to a DataFrame with column names (fixes the warning!)
    fake_houses = pd.DataFrame([
        [8.0, 20.0, 6.0, 1.0, 300.0, 3.0, 34.05, -118.25],
        [2.0, 50.0, 4.0, 1.2, 500.0, 4.0, 38.50, -121.40],
        [5.0, 10.0, 5.0, 1.0, 200.0, 2.5, 37.77, -122.42]
    ], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    
    predictions = model.predict(fake_houses)
    
    print("\n--- Example Predictions ---")
    for i, pred in enumerate(predictions):
        price_in_dollars = pred * 100000 
        print(f"Fake House {i+1} Predicted Price: ${price_in_dollars:,.2f}")

if __name__ == "__main__":
    make_sample_predictions()