# 🏠 House Price Prediction (California Housing)

A complete, end-to-end Machine Learning project that predicts median house prices in California based on features like median income, house age, and location.

This project demonstrates the standard ML pipeline: Data Loading, Feature Exploration, Model Training, Evaluation, and Deployment (Saving/Loading the model).

---

## 📊 Dataset

This project uses the built-in **California Housing Dataset** provided by `scikit-learn`.

It contains **20,640 housing blocks** with 8 numerical features such as:

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`

> **Note:** The target variable (`Price`) is expressed in hundreds of thousands of dollars (`$100,000`).

---

## 🗂️ Project Structure

```text
House Price Prediction/
│
├── data/                   # Directory for datasets
│
├── models/                 # Contains the saved .pkl model file
│   └── house_price_model.pkl
│
├── notebooks/              # Jupyter notebooks for data exploration
│   └── 01_exploration.ipynb
│
├── src/                    # Source code for the ML pipeline
│   ├── __init__.py
│   ├── data_processing.py  # Loads and cleans the dataset
│   ├── model.py            # Trains and evaluates the model
│   └── predict.py          # Makes predictions using the saved model
│
├── main.py                 # Runs the entire pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Methodology

The project follows these 4 main steps:

### 1️⃣ Data Processing
- Load the California housing dataset
- Check for missing values
- Prepare features and target variables

### 2️⃣ Model Training
- Split the dataset into:
  - **80% Training Data**
  - **20% Testing Data**
- Train a `LinearRegression` model

### 3️⃣ Evaluation
Evaluate model performance using:

- **Root Mean Squared Error (RMSE)**
- **R-squared Score (R²)**

Also analyze feature importance using model coefficients.

### 4️⃣ Deployment
- Save the trained model using `joblib`
- Load the model later for predicting new house prices

---

## 🚀 How to Run This Project

### ✅ Prerequisites

Make sure you have:

- Python 3 installed
- `pip` installed

---

### ⚙️ Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/Exnlab/House-Price-Prediction.git
cd House-Price-Prediction
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the Project

```bash
python main.py
```

---

## 📈 Results

After training the Linear Regression model, the following results are achieved on the test dataset:

| Metric | Score |
|--------|--------|
| R² Score | ~0.575 |
| RMSE | ~0.745 |

### Interpretation
- The model explains approximately **57.5%** of the variance in house prices.
- The average prediction error is around **$74,500**.

---

## 🔮 Future Improvements

Some possible improvements for this project:

- Build a **Streamlit** or **Flask** web application
- Use advanced ML models like:
  - Random Forest
  - XGBoost
  - Gradient Boosting
- Add data visualizations:
  - Heatmaps
  - Feature distributions
  - Correlation analysis
- Perform hyperparameter tuning for better accuracy

---

## 🧠 Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Matplotlib
- Jupyter Notebook

---

## ⭐ Project Goal

This project is designed for beginners who want to understand the **complete Machine Learning workflow** using a real-world dataset.

It covers:
- Data preprocessing
- Model training
- Evaluation
- Saving/loading ML models
- Prediction pipeline

---
