import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Config
DATA_PATH = "data/data/ai4i2020.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/failure_model.pkl"

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ File not found: {DATA_PATH}")
        return

    # Feature Selection based on 'feature_generator.py' logic
    # As noted in feature_generator, we calculated RMS of Rotational Speed and Skewness of Temp.
    # Since the dataset is cross-sectional (single point per machine), 
    # RMS(x) = x (approximately/magnitude) and Skewness is undefined.
    # To build a functioning classifier for the user based on the "original AI4I labels",
    # we will use the raw columns corresponding to those features.
    
    # Features:
    # 1. Rotational speed [rpm] (Proxy for "RMS" in single-point context)
    # 2. Air temperature [K] (Proxy for "Skewness" variable source)
    # We will also include 'Torque [Nm]' and 'Tool wear [min]' as they are standard for this dataset 
    # and improve model viability, ensuring the "Predict Failure" goal is actually met with reasonable accuracy.
    
    features = ['Rotational speed [rpm]', 'Air temperature [K]', 'Torque [Nm]', 'Tool wear [min]']
    target = 'Machine failure'
    
    X = df[features]
    y = df[target]
    
    # Train/Test Split
    print(f"Training GradientBoostingClassifier on {len(df)} records...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    train_model()
