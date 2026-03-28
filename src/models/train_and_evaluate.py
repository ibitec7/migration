import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.models.surge_metrics import evaluate_surge_performance
from src.models.surge_model import MigrationLSTM, SurgeJointLoss, build_sequential_tensors

# Model directories
MODELS_DIR = Path("src/models/trained_models")
PLOTS_DIR = Path("data/plots/model_performance")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Prepare Data
df = pl.read_parquet("data/processed/train_panel.parquet").drop_nulls()
train_df = df.filter(pl.col("month").dt.year() <= 2022)
test_df = df.filter(pl.col("month").dt.year() > 2022)

feature_cols = [c for c in df.columns if "lag" in c]
target_cols = [f"target_visa_lead_{i}" for i in range(1, 7)]

X_train_tab = train_df.select(feature_cols).to_numpy()
y_train_tab = train_df.select(target_cols).to_numpy()
X_test_tab = test_df.select(feature_cols).to_numpy()
y_test_tab = test_df.select(target_cols).to_numpy()

print("\n--- Training & Saving Random Forest Baseline ---")
rf_models = []
rf_preds = []
for i in range(6):
    rf = RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42)
    rf.fit(X_train_tab, y_train_tab[:, i])
    rf_preds.append(rf.predict(X_test_tab))
    joblib.dump(rf, MODELS_DIR / f"rf_lead_{i+1}.joblib")
    rf_models.append(rf)
rf_preds = np.stack(rf_preds, axis=1)

# 2. DL Seq Models Data using the existing builder
X_train_seq, Y_train_seq, C_train_seq, scaler_x, scaler_y, num_c = build_sequential_tensors(train_df, is_train=True)
X_test_seq, Y_test_seq, C_test_seq, _, _, _ = build_sequential_tensors(test_df, scaler_x=scaler_x, scaler_y=scaler_y, is_train=False)

joblib.dump(scaler_x, MODELS_DIR / "scaler_x.joblib")
joblib.dump(scaler_y, MODELS_DIR / "scaler_y.joblib")

# Also save the mapping
unique_countries = sorted(train_df["country"].unique().to_list())
with open(MODELS_DIR / "country_map.json", "w") as f:
    json.dump({name: idx for idx, name in enumerate(unique_countries)}, f)

train_dataset = TensorDataset(X_train_seq, Y_train_seq, C_train_seq)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transformer Model
class MigrationTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=2, output_dim=6):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 6, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x, c=None):
        # Ignored country embedding for pure transformer baseline
        x = self.embedding(x) + self.pos_encoder
        out = self.transformer(x)
        return self.fc(out[:, -1, :])

print("\n--- Training & Saving Transformer ---")
tf_model = MigrationTransformer().to(device)
opt_tf = torch.optim.Adam(tf_model.parameters(), lr=1e-3)
crit_tf = nn.MSELoss()
tf_model.train()
for epoch in range(20):
    for bx, by, _ in train_loader:
        bx, by = bx.to(device), by.to(device)
        opt_tf.zero_grad()
        out = tf_model(bx)
        loss = crit_tf(out, by)
        loss.backward()
        opt_tf.step()

torch.save(tf_model.state_dict(), MODELS_DIR / "transformer.pth")

tf_model.eval()
with torch.no_grad():
    tf_preds_scaled = tf_model(X_test_seq.to(device)).cpu().numpy()
    tf_preds = scaler_y.inverse_transform(tf_preds_scaled)

print("\n--- Training & Saving LSTM (SurgeJointLoss) ---")
lstm_model = MigrationLSTM(num_countries=num_c).to(device)
opt_lstm = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
crit_lstm = SurgeJointLoss(alpha=0.6, surge_threshold=1.5)

lstm_model.train()
for epoch in range(25):
    for bx, by, bc in train_loader:
        bx, by, bc = bx.to(device), by.to(device), bc.to(device)
        opt_lstm.zero_grad()
        out = lstm_model(bx, bc)
        loss = crit_lstm(out, by)
        loss.backward()
        opt_lstm.step()

torch.save(lstm_model.state_dict(), MODELS_DIR / "lstm.pth")

lstm_model.eval()
with torch.no_grad():
    lstm_preds_scaled = lstm_model(X_test_seq.to(device), C_test_seq.to(device)).cpu().numpy()
    lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled)

# 3. Horizon-Aware Ensemble
print("\n--- Generating Horizon-Aware Ensemble ---")
# Weights dynamically assign focus depending on the lead time based on benchmark strengths
w_rf = np.array([0.60, 0.50, 0.30, 0.20, 0.10, 0.05])
w_tf = np.array([0.10, 0.20, 0.40, 0.60, 0.80, 0.85])
w_ls = np.array([0.30, 0.30, 0.30, 0.20, 0.10, 0.10])

# Normalize weights securely
totals = w_rf + w_tf + w_ls
w_rf, w_tf, w_ls = w_rf / totals, w_tf / totals, w_ls / totals

ens_preds = np.zeros_like(y_test_tab)
for i in range(6):
    ens_preds[:, i] = (
        w_rf[i] * rf_preds[:, i] + 
        w_tf[i] * tf_preds[:, i] + 
        w_ls[i] * lstm_preds[:, i]
    )

# 4. Evaluation & Plotting 
results = {
    'RF': {'f1': [], 'prec': [], 'rec': []},
    'Transformer': {'f1': [], 'prec': [], 'rec': []},
    'LSTM': {'f1': [], 'prec': [], 'rec': []},
    'Ensemble': {'f1': [], 'prec': [], 'rec': []}
}

models_preds = {
    'RF': rf_preds,
    'Transformer': tf_preds,
    'LSTM': lstm_preds,
    'Ensemble': ens_preds
}

for m_name, preds in models_preds.items():
    print(f"\n[{m_name} Surge Classification]")
    for i in range(6):
        sm = evaluate_surge_performance(y_test_tab[:, i], preds[:, i], threshold_std=1.5)
        print(f"Lead {i+1} | Prec: {sm['precision']:.2f}, Rec: {sm['recall']:.2f}, F1: {sm['f1']:.2f}")
        results[m_name]['f1'].append(sm['f1'])
        results[m_name]['prec'].append(sm['precision'])
        results[m_name]['rec'].append(sm['recall'])

# Let's plot F1 Score
plt.figure(figsize=(10, 6))
lags = np.arange(1, 7)
plt.plot(lags, results['RF']['f1'], marker='o', label='cuML RF Base')
plt.plot(lags, results['Transformer']['f1'], marker='s', label='Transformer')
plt.plot(lags, results['LSTM']['f1'], marker='^', label='LSTM (JointLoss)')
plt.plot(lags, results['Ensemble']['f1'], marker='*', markersize=12, linewidth=3, label='Horizon-Aware Ensemble')

plt.title('F1-Score for Crisis Surge Detection (>1.5 Std Dev)')
plt.xlabel('Lead Time (Months)')
plt.ylabel('F1 Score')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'f1_score_by_horizon.png', dpi=300)
plt.close()

# Plot Precision-Recall Tradeoffs as bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bar_width = 0.2
x = np.arange(6)
ax1.bar(x - 1.5*bar_width, results['RF']['prec'], bar_width, label='RF')
ax1.bar(x - 0.5*bar_width, results['Transformer']['prec'], bar_width, label='Transformer')
ax1.bar(x + 0.5*bar_width, results['LSTM']['prec'], bar_width, label='LSTM')
ax1.bar(x + 1.5*bar_width, results['Ensemble']['prec'], bar_width, label='Ensemble')
ax1.set_title("Precision by Model")
ax1.set_xticks(x, [f"Lead {i}" for i in range(1, 7)])
ax1.legend()

ax2.bar(x - 1.5*bar_width, results['RF']['rec'], bar_width, label='RF')
ax2.bar(x - 0.5*bar_width, results['Transformer']['rec'], bar_width, label='Transformer')
ax2.bar(x + 0.5*bar_width, results['LSTM']['rec'], bar_width, label='LSTM')
ax2.bar(x + 1.5*bar_width, results['Ensemble']['rec'], bar_width, label='Ensemble')
ax2.set_title("Recall by Model")
ax2.set_xticks(x, [f"Lead {i}" for i in range(1, 7)])

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'precision_recall_comparison.png', dpi=300)
plt.close()

print("\nModels successfully saved to 'src/models/trained_models/'")
print("Performance plots saved to 'data/plots/model_performance/'")

