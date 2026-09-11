"""
River Water Quality Analysis: Tabular ML & POV-Ray Pipeline
-----------------------------------------------------------
Execution:
- Open this file in Python IDLE and press F5 (Run -> Run Module).
- Or run from terminal: python run_pipeline.py
"""

import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# 1. PATH SETUP & DATA INGESTION
# ---------------------------------------------------------
print("=" * 60)
print("STAGE 1: LOADING & AUDITING EXPERIMENTAL DATA")
print("=" * 60)

candidate_paths = [
    'Reproducible_Dataset.xlsx',
    os.path.join('data', 'raw', 'Reproducible_Dataset.xlsx'),
    os.path.join('..', 'data', 'raw', 'Reproducible_Dataset.xlsx'),
    os.path.join('..', 'Reproducible_Dataset.xlsx')
]

dataset_path = None
for p in candidate_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    print("ERROR: 'Reproducible_Dataset.xlsx' was not found.")
    print("Please place 'Reproducible_Dataset.xlsx' in the script folder.")
    sys.exit(1)

print(f"Loading data from: {dataset_path}")
df = pd.read_excel(dataset_path, sheet_name='Raw_Data')

features = ['Alkalinity', 'BOD', 'COD', 'Chloride', 'DO', 'EC', 'PH', 'SS', 'TDS', 'Turbidity']
X = df[features]
y = df['WQI']
y_class = df['WQI_Class']

print(f"Dataset verified: {X.shape[0]} observations, {X.shape[1]} physical features.")
print(f"Missing values across features: {X.isnull().sum().sum()}")

# Create organized output directories
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'povray_scenes'), exist_ok=True)
os.makedirs(os.path.join('data', 'processed'), exist_ok=True)

# ---------------------------------------------------------
# 2. GRADIENT BOOSTED TREE REGRESSION
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STAGE 2: TRAINING GRADIENT BOOSTED TREES (REGRESSION)")
print("=" * 60)

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_all = scaler.fit_transform(X)

# Model initialization (Histogram-based GBDT)
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.03,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluation on holdout test set
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test Set R² Score: {r2:.4f}")
print(f"Test Set MAE:      {mae:.4f} WQI units")
print(f"Test Set RMSE:     {rmse:.4f} WQI units")

# Permutation Feature Importance
perm = permutation_importance(model, X_test_scaled, y_test, n_repeats=20, random_state=42)
perm_df = pd.DataFrame({
    'feature': features,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nPermutation Feature Importance Rankings (Drop in R²):")
for idx, row in perm_df.iterrows():
    print(f"  {row['feature']:<12}: {row['importance_mean']:.4f} +/- {row['importance_std']:.4f}")

# ---------------------------------------------------------
# 3. 3D MANIFOLD PROJECTION & POV-RAY SCENE GENERATION
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STAGE 3: 3D MANIFOLD LEARNING & POV-RAY SCENE EXPORT")
print("=" * 60)

print("Computing 3D non-linear manifold projection...")
manifold = TSNE(n_components=3, random_state=42, perplexity=30)
coords_3d = manifold.fit_transform(X_scaled_all)

# Center and scale to standard [-5, 5] POV-Ray bounding box
coords_3d -= coords_3d.mean(axis=0)
coords_3d /= np.abs(coords_3d).max()
coords_3d *= 5.0

# Export processed 3D coordinates
df_coords = pd.DataFrame(coords_3d, columns=['X', 'Y', 'Z'])
df_coords['WQI'] = y
df_coords['WQI_Class'] = y_class
df_coords['River'] = df['River']
coords_csv = os.path.join('data', 'processed', 'manifold_embeddings_3d.csv')
df_coords.to_csv(coords_csv, index=False)
print(f"Saved 3D coordinates: {coords_csv}")

# Write POV-Ray Scene Description Language (.pov)
pov_path = os.path.join('outputs', 'povray_scenes', 'water_manifold.pov')
color_map_pov = {
    'Good': 'rgb <0.18, 0.70, 0.25>',
    'Poor': 'rgb <0.20, 0.50, 0.85>',
    'Very Poor': 'rgb <1.00, 0.55, 0.10>',
    'Unsuitable for Drinking': 'rgb <0.85, 0.15, 0.15>'
}

with open(pov_path, 'w') as f:
    f.write("""// POV-Ray Ray-Traced 3D Manifold Scene
#version 3.7;
global_settings { assumed_gamma 1.0 }

// Studio Camera Configuration
camera {
    location <0, 8, -14>
    look_at <0, 0, 0>
    angle 45
}

// Three-Point Studio Lighting
light_source { <15, 20, -15> color rgb <1.0, 1.0, 1.0> } // Key Light
light_source { <-15, 10, -10> color rgb <0.4, 0.4, 0.4> } // Fill Light
light_source { <0, -10, 5>   color rgb <0.2, 0.2, 0.2> } // Rim Light

background { color rgb <0.96, 0.96, 0.98> }

// Bounding Reference Box
box {
    <-5.5, -5.5, -5.5>, <5.5, 5.5, 5.5>
    pigment { color rgbt <0.75, 0.75, 0.80, 0.88> }
    finish { phong 0.1 }
}
""")
    for (x_pt, y_pt, z_pt), cls in zip(coords_3d, y_class):
        pigment = color_map_pov.get(cls, 'rgb <0.5, 0.5, 0.5>')
        f.write(f"sphere {{ <{x_pt:.4f}, {y_pt:.4f}, {z_pt:.4f}>, 0.18 pigment {{ {pigment} }} finish {{ specular 0.6 roughness 0.02 reflection 0.04 }} }}\n")

print(f"Generated POV-Ray scene file: {pov_path}")

# Check for local POV-Ray binary
povray_bin = shutil.which("povray")
if povray_bin:
    print(f"Found POV-Ray at '{povray_bin}'. Rendering photorealistic scene...")
    png_out = os.path.join('outputs', 'povray_scenes', 'water_manifold.png')
    subprocess.run([povray_bin, "+W1280", "+H720", "+A0.1", "+Q11", "+FN", pov_path, f"-O{png_out}"])
    print(f"Rendered: {png_out}")
else:
    print("NOTE: 'povray' command-line binary not detected on system PATH.")
    print(f"Scene is saved at '{pov_path}' and ready to open in POV-Ray Windows GUI.")

# ---------------------------------------------------------
# 4. DIAGNOSTIC FIGURES (MATPLOTLIB / SEABORN)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STAGE 4: GENERATING & DISPLAYING DIAGNOSTIC CHARTS")
print("=" * 60)

fig = plt.figure(figsize=(16, 7))

# Panel 1: 3D Projection
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
colors_mpl = {
    'Good': '#2ca02c',
    'Poor': '#1f77b4',
    'Very Poor': '#ff7f0e',
    'Unsuitable for Drinking': '#d62728'
}

for cls in ['Good', 'Poor', 'Very Poor', 'Unsuitable for Drinking']:
    mask = (df_coords['WQI_Class'] == cls)
    ax1.scatter(
        df_coords.loc[mask, 'X'],
        df_coords.loc[mask, 'Y'],
        df_coords.loc[mask, 'Z'],
        c=colors_mpl[cls],
        label=cls,
        s=35,
        alpha=0.85,
        edgecolors='k',
        linewidth=0.3
    )

ax1.set_xlim([-5.5, 5.5])
ax1.set_ylim([-5.5, 5.5])
ax1.set_zlim([-5.5, 5.5])
ax1.set_xlabel('Manifold X', labelpad=10)
ax1.set_ylabel('Manifold Y', labelpad=10)
ax1.set_zlabel('Manifold Z', labelpad=10)
ax1.view_init(elev=25, azim=-60)
ax1.set_title('3D Manifold Simulation (POV-Ray Camera Perspective)', fontsize=12, fontweight='bold')
ax1.legend(title='WQI Status', loc='upper left')

# Panel 2: Observed vs Predicted WQI
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(y_test, y_pred, color='#2b5c8f', edgecolors='k', s=55, alpha=0.75, label='Holdout Test Observations')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Ideal Fit Line')
ax2.set_xlabel('Actual Water Quality Index (WQI)', fontsize=11)
ax2.set_ylabel('Predicted WQI (Gradient Boosted Trees)', fontsize=11)
ax2.set_title(f'Regression Agreement (R² = {r2:.4f}, RMSE = {rmse:.2f})', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig_out1 = os.path.join('outputs', 'figures', '3d_manifold_and_validation.png')
plt.savefig(fig_out1, dpi=300)
print(f"Saved figure: {fig_out1}")

# Feature Importance & Residual Distribution
fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(data=perm_df, x='importance_mean', y='feature', ax=axes[0], palette='Blues_r', edgecolor='k')
axes[0].errorbar(perm_df['importance_mean'], range(len(perm_df)), xerr=perm_df['importance_std'], fmt='none', c='black', capsize=4)
axes[0].set_title('Permutation Feature Importance (Test Set)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Mean Drop in R²')
axes[0].set_ylabel('Parameter')
axes[0].grid(axis='x', linestyle=':', alpha=0.6)

residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=axes[1], color='#1f77b4', edgecolor='k', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
axes[1].set_title('Residual Error Distribution (Actual - Predicted)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residual (WQI Units)')
axes[1].set_ylabel('Count')
axes[1].grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
fig_out2 = os.path.join('outputs', 'figures', 'feature_importance_and_residuals.png')
plt.savefig(fig_out2, dpi=300)
print(f"Saved figure: {fig_out2}")

print("\n" + "=" * 60)
print("PIPELINE EXECUTION COMPLETE! Interactive plots opened.")
print("=" * 60)
plt.show()
