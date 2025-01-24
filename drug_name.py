# drug_name.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

class DrugModelTrainer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.vectorizer = None
        self.data = None
        
    def load_data(self):
        """Load and preprocess the CSV data."""
        try:
            self.data = pd.read_csv(self.csv_path, encoding='latin1')
            self.data['Medicine List'] = self.data['Medicine Names'].apply(
                lambda x: [item.strip() for item in x.split(',')]
            )
            self.data['Medicine Count'] = self.data['Medicine List'].apply(len)
            self.aggregated_data = self.data.loc[
                self.data.groupby('Condition')['Medicine Count'].idxmax()
            ]
            self.aggregated_data['Medicine Names'] = self.aggregated_data['Medicine List'].apply(', '.join)
            return True
        except Exception as e:
            print(f"Error loading CSV data: {str(e)}")
            return False
            
    def train_model(self):
        """Train the model using the loaded data."""
        try:
            # Prepare the data
            X = self.aggregated_data['Medicine Names']
            y = self.aggregated_data['Condition']
            
            # Initialize and fit the vectorizer
            self.vectorizer = TfidfVectorizer()
            X_tfidf = self.vectorizer.fit_transform(X)
            
            # Train the model
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X_tfidf, y)
            print("Model trained successfully.")
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
            
    def save_model(self, model_path='medicine_condition_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        """Save the trained model and vectorizer."""
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            print(f"Model saved to {model_path}")
            print(f"Vectorizer saved to {vectorizer_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
            
    def train_and_save(self):
        """Complete pipeline to load data, train and save the model."""
        if self.load_data() and self.train_model():
            return self.save_model()
        return False

class DrugModelPredictor:
    def __init__(self, model_path='medicine_condition_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
    def predict_condition(self, medicines):
        """Predict condition for given medicines."""
        if isinstance(medicines, str):
            medicines = [medicines]
        medicines_text = ", ".join(medicines)
        features = self.vectorizer.transform([medicines_text])
        return self.model.predict(features)[0]
        
    def get_condition_probability(self, medicines):
        """Get probability scores for all conditions."""
        if isinstance(medicines, str):
            medicines = [medicines]
        medicines_text = ", ".join(medicines)
        features = self.vectorizer.transform([medicines_text])
        probabilities = self.model.predict_proba(features)[0]
        conditions = self.model.classes_
        return dict(zip(conditions, probabilities))

def main():
    # Set the path to your CSV file
    csv_path = "E:/medicine_name.csv"  # Update this path
    
    # Create and train the model
    trainer = DrugModelTrainer(csv_path)
    if trainer.train_and_save():
        print("Model training and saving completed successfully.")
        
        # Test the model
        predictor = DrugModelPredictor()
        test_medicines = ["Paracetamol", "Ibuprofen"]
        
        # Make predictions
        condition = predictor.predict_condition(test_medicines)
        probabilities = predictor.get_condition_probability(test_medicines)
        
        print(f"\nTest Results:")
        print(f"Medicines: {', '.join(test_medicines)}")
        print(f"Predicted Condition: {condition}")
        print("\nProbabilities for each condition:")
        for cond, prob in probabilities.items():
            print(f"{cond}: {prob:.2%}")

if __name__ == "__main__":
    main()