from src.data_processing import get_data
from src.model import train_and_evaluate
from src.predict import make_sample_predictions

if __name__ == "__main__":
    print("========================================")
    print("  HOUSE PRICE PREDICTION PIPELINE")
    print("========================================\n")
    
    # This will load data, train, evaluate, save the model, AND predict - all at once!
    train_and_evaluate()
    make_sample_predictions()
    
    print("\n========================================")
    print("  PIPELINE FINISHED SUCCESSFULLY!")
    print("========================================")