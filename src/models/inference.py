import torch
import joblib
import json
import numpy as np
from pathlib import Path
from src.models.surge_model import MigrationLSTM, MigrationTransformer


class MigrationSurgeEnsemble:
    def __init__(self, models_dir="src/models/trained_models"):
        self.models_dir = Path(models_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load mappings and scalers
        with open(self.models_dir / "country_map.json", "r") as f:
            self.country_map = json.load(f)
            
        self.scaler_x = joblib.load(self.models_dir / "scaler_x.joblib")
        self.scaler_y = joblib.load(self.models_dir / "scaler_y.joblib")
        
        # Load cuML / Sklearn Random Forests (1 per lead month)
        self.rf_models = []
        for i in range(6):
            self.rf_models.append(joblib.load(self.models_dir / f"rf_lead_{i+1}.joblib"))
            
        # Load Deep Learning Models
        self.lstm = MigrationLSTM(num_countries=len(self.country_map)).to(self.device)
        self.lstm.load_state_dict(torch.load(self.models_dir / "lstm.pth", map_location=self.device))
        self.lstm.eval()
        
        self.transformer = MigrationTransformer().to(self.device)
        self.transformer.load_state_dict(torch.load(self.models_dir / "transformer.pth", map_location=self.device))
        self.transformer.eval()
        
        # Define the proven horizon-aware ensemble weights
        w_rf = np.array([0.60, 0.50, 0.30, 0.20, 0.10, 0.05])
        w_tf = np.array([0.10, 0.20, 0.40, 0.60, 0.80, 0.85])
        w_ls = np.array([0.30, 0.30, 0.30, 0.20, 0.10, 0.10])

        totals = w_rf + w_tf + w_ls
        self.w_rf = w_rf / totals
        self.w_tf = w_tf / totals
        self.w_ls = w_ls / totals

    def predict(self, country_name, recent_6_months_data):
        """
        Expects a single country name, and a list of 6 lists representing rolling history (Lag 6 to Lag 1)
        where each row has [visa_volume, exchange_rate, news_sentiment_count].
        Returns predicted raw volume and binary flags indicating surges across 6 future months.
        """
        if country_name not in self.country_map:
            raise ValueError(f"Country {country_name} not found in training data.")
        
        c_id = self.country_map[country_name]
        
        # Convert to expected shapes
        # Random forest expects flat (1, 18)
        # Deep learning expects seq (1, 6, 3)
        features_2d = np.array(recent_6_months_data).astype(np.float32) # Shape: (6, 3)
        features_flat = features_2d.flatten().reshape(1, -1)   # Shape: (1, 18)
        
        # Apply scaling
        x_scaled_flat = self.scaler_x.transform(features_flat)
        x_seq_scaled = torch.tensor(x_scaled_flat.reshape(1, 6, 3), dtype=torch.float32).to(self.device)
        c_tensor = torch.tensor([c_id], dtype=torch.long).to(self.device)
        
        # 1. Random Forest Predictions
        rf_preds_scaled = np.zeros((1, 6))
        for i in range(6):
            rf_preds_scaled[0, i] = self.rf_models[i].predict(x_scaled_flat)[0]
            
        # 2. LSTM & Transformer Predictions
        with torch.no_grad():
            lstm_preds_scaled = self.lstm(x_seq_scaled, c_tensor).cpu().numpy()
            tf_preds_scaled = self.transformer(x_seq_scaled, c_tensor).cpu().numpy()
            
        # 3. Dynamic Horizon Ensemble Blending
        ensemble_preds_scaled = np.zeros_like(rf_preds_scaled)
        for i in range(6):
            ensemble_preds_scaled[0, i] = (
                self.w_rf[i] * rf_preds_scaled[0, i] + 
                self.w_tf[i] * tf_preds_scaled[0, i] + 
                self.w_ls[i] * lstm_preds_scaled[0, i]
            )
            
        # 4. Inverse Scale back to Real Monthly Volumes
        rf_preds = self.scaler_y.inverse_transform(rf_preds_scaled)[0]
        lstm_preds = self.scaler_y.inverse_transform(lstm_preds_scaled)[0]
        tf_preds = self.scaler_y.inverse_transform(tf_preds_scaled)[0]
        final_preds = self.scaler_y.inverse_transform(ensemble_preds_scaled)[0]
        
        return {
            'Horizon (Months Ahead)': [1, 2, 3, 4, 5, 6],
            'Ensemble Prediction Volume': final_preds.round(0).tolist(),
            'Individual Model Volumes': {
                'RandomForest': rf_preds.round(0).tolist(),
                'LSTM': lstm_preds.round(0).tolist(),
                'Transformer': tf_preds.round(0).tolist()
            }
        }

if __name__ == "__main__":
    # Example Usage on dummy inference data mimicking a crisis trajectory
    print("\nInitializing Horizon-Aware Ensemble Forecaster...\n")
    predictor = MigrationSurgeEnsemble()
    
    # E.g. Mexico with a hypothetical aggressive 6-month buildup
    # Format: [visa_volume, exchange_rate, news_sentiment_count]
    historical_scenario = [
        [15000, 19.5, 45], # T-6
        [16000, 19.8, 52], # T-5
        [18500, 19.9, 70], # T-4
        [22000, 20.3, 85], # T-3
        [24000, 20.5, 110],# T-2
        [31000, 21.0, 140] # T-1 (Current Month showing huge crisis buildup)
    ]
    
    country = "Mexico"
    print(f"Predicting next 6 months for {country} based on recent massive buildup...")
    
    results = predictor.predict(country, historical_scenario)
    
    print("-" * 50)
    print(f"{'Lead Month':<15} | {'Ensemble Target Volume'}")
    print("-" * 50)
    
    for lead, vol in zip(results['Horizon (Months Ahead)'], results['Ensemble Prediction Volume']):
        print(f"Lead {lead:<10} | {int(vol):,} projected individuals")
        
    print("\nUnderlying Model Variations:")
    for m in ['RandomForest', 'LSTM', 'Transformer']:
        print(f"{m:<13}: {results['Individual Model Volumes'][m]}")

