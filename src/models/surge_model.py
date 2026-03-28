import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.surge_metrics import evaluate_surge_performance

class SurgeJointLoss(nn.Module):
    def __init__(self, alpha=0.5, surge_threshold=1.0):
        """
        Huber Loss for volume + BCE for surges.
        If actual target is > threshold standard deviations above the mean, 
        it heavily penalizes missing it.
        """
        super().__init__()
        self.huber = nn.HuberLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.threshold = surge_threshold

    def forward(self, pred_vol, target_vol):
        # Continuous Volume Loss
        l_vol = self.huber(pred_vol, target_vol)
        
        # Determine Surge implicitly (1.0 = 1 standard deviation for standardized targets)
        # Assuming our targets are StandardScaled before passing to PyTorch
        true_surge_flags = (target_vol > self.threshold).float()
        
        # We'll use the predicted continuous volume and threshold it inside a Sigmoid pseudo-range
        pred_surge_logits = pred_vol - self.threshold
        l_surge = self.bce(pred_surge_logits, true_surge_flags)
        
        return self.alpha * l_vol + (1 - self.alpha) * l_surge

class MigrationLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=6, num_countries=15):
        super().__init__()
        # Country Embedding: Map discrete country IDs to dense vectors
        self.country_emb = nn.Embedding(num_countries, 8)
        
        # We append country_emb to the sequenced features
        self.lstm = nn.LSTM(input_dim + 8, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, c_idx):
        # x is (B, Seq, F)
        # c_idx is (B,)
        c = self.country_emb(c_idx)  # (B, 8)
        # Expand and concat across sequence
        c = c.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, Seq, 8)
        
        x_combined = torch.cat([x, c], dim=-1)  # (B, Seq, F+8)
        out, _ = self.lstm(x_combined)
        return self.fc(out[:, -1, :])

def build_sequential_tensors(df_pl, scaler_x=None, scaler_y=None, is_train=True):
    # Determine distinct countries list based on sorted unique
    unique_countries = sorted(df_pl["country"].unique().to_list())
    country_map = {name: idx for idx, name in enumerate(unique_countries)}
    
    # Store mapped IDs to columns natively
    df_pl = df_pl.with_columns([
        pl.col("country").replace(country_map).cast(pl.Int64).alias("country_id")
    ])
    
    country_ids = df_pl["country_id"].to_numpy()
    
    seq_features = []
    for lag in range(6, 0, -1):
        lag_cols = [f"visa_lag_{lag}", f"exchange_lag_{lag}", f"news_lag_{lag}"]
        seq_features.append(df_pl.select(lag_cols).to_numpy())
        
    X_seq = np.stack(seq_features, axis=1)
    
    target_cols = [f"target_visa_lead_{i}" for i in range(1, 7)]
    Y = df_pl.select(target_cols).to_numpy()

    N, seq_len, f_dim = X_seq.shape
    X_flat = X_seq.reshape(-1, f_dim)
    
    if is_train:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_flat_scaled = scaler_x.fit_transform(X_flat)
        Y_scaled = scaler_y.fit_transform(Y)
    else:
        X_flat_scaled = scaler_x.transform(X_flat)
        Y_scaled = scaler_y.transform(Y)

    X_seq_scaled = X_flat_scaled.reshape(N, seq_len, f_dim)
    
    return (
        torch.tensor(X_seq_scaled, dtype=torch.float32), 
        torch.tensor(Y_scaled, dtype=torch.float32),
        torch.tensor(country_ids, dtype=torch.long),
        scaler_x, 
        scaler_y,
        len(unique_countries)
    )

def train_surge_dl(model, train_loader, epochs=20, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Using our custom SurgeJointLoss which targets RMSE on normal + BCE for >1.0 std spikes
    criterion = SurgeJointLoss(alpha=0.6, surge_threshold=1.5) 
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y, batch_c in train_loader:
            batch_x, batch_y, batch_c = batch_x.to(device), batch_y.to(device), batch_c.to(device)
            optimizer.zero_grad()
            out = model(batch_x, batch_c)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    return model

def run_evaluation_split():
    df = pl.read_parquet("data/processed/train_panel.parquet").drop_nulls()

    train_df = df.filter(pl.col("month").dt.year() <= 2022)
    test_df = df.filter(pl.col("month").dt.year() > 2022)

    X_train_seq, Y_train_seq, C_train_seq, scaler_x, scaler_y, num_c = build_sequential_tensors(train_df, is_train=True)
    X_test_seq, Y_test_seq, C_test_seq, _, _, _ = build_sequential_tensors(test_df, scaler_x=scaler_x, scaler_y=scaler_y, is_train=False)

    batch_size = 256
    train_dataset = TensorDataset(X_train_seq, Y_train_seq, C_train_seq)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train LSTM with Country Embeddings and SurgeJointLoss
    lstm_model = MigrationLSTM(num_countries=num_c)
    lstm_model = train_surge_dl(lstm_model, train_loader, epochs=25, lr=1e-3, device=device)
    
    lstm_model.eval()
    with torch.no_grad():
        lstm_preds_scaled = lstm_model(X_test_seq.to(device), C_test_seq.to(device)).cpu().numpy()
        lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled)
        y_test_unscaled = scaler_y.inverse_transform(Y_test_seq.numpy())

    print("\n[Surge Target Discovery Results - PyTorch LSTM + Huber/BCE]")
    # Test surge capabilities specifically
    for i in range(6):
        truth = y_test_unscaled[:, i]
        pred = lstm_preds[:, i]
        
        rmse = mean_squared_error(truth, pred) ** 0.5
        surge_metrics = evaluate_surge_performance(truth, pred, threshold_std=1.5)
        
        print(f"Lead {i+1} Month | RMSE: {rmse:.2f} | " + 
               f"Surge Precision: {surge_metrics['precision']:.2f}, " +
               f"Recall: {surge_metrics['recall']:.2f}, " +
               f"F1: {surge_metrics['f1']:.2f} (Total Surges: {surge_metrics['surges_found']})")

if __name__ == "__main__":
    run_evaluation_split()

class MigrationTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=2, output_dim=6):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 6, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x, c=None):
        # We allow `c` to be passed to match predicting signatures, even if ignored
        x = self.embedding(x) + self.pos_encoder
        out = self.transformer(x)
        return self.fc(out[:, -1, :])
