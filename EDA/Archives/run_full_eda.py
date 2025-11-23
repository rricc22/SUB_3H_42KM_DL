#!/usr/bin/env python3
"""
Complete EDA execution script - runs all analysis and saves results
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("="*80)
print("RUNNING COMPLETE EDA AND BASELINE MODEL")
print("="*80)

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Load data
print("\n1. Loading 1000 workouts...")
def load_endomondo_sample(filepath, n_samples=1000):
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            try:
                workout = ast.literal_eval(line.strip())
                data.append(workout)
            except:
                continue
    return data

workouts = load_endomondo_sample('endomondoHR.json', n_samples=1000)
print(f"   Loaded {len(workouts)} workouts")

# 2. Sport distribution
sports = [w['sport'] for w in workouts]
sport_counts = pd.Series(sports).value_counts()
print("\n2. Sport Distribution:")
print(sport_counts.head(10))

# 3. Create labels
def create_binary_labels(workouts):
    labels = []
    for w in workouts:
        sport = w['sport'].lower()
        if 'run' in sport:
            labels.append(0)  # Genuine
        else:
            labels.append(1)  # Fraud
    return np.array(labels)

labels = create_binary_labels(workouts)
print(f"\n3. Labels:")
print(f"   Genuine runs: {np.sum(labels == 0)}")
print(f"   Fraudulent (bike): {np.sum(labels == 1)}")

# 4. Extract features
print("\n4. Extracting features...")

def extract_features(workout):
    features = {}
    
    # Speed features
    if 'speed' in workout and workout['speed'] is not None:
        speed = np.array(workout['speed'], dtype=float)
        speed = speed[~np.isnan(speed)]
        if len(speed) > 0:
            features['speed_mean'] = np.mean(speed)
            features['speed_std'] = np.std(speed)
            features['speed_max'] = np.max(speed)
            features['speed_min'] = np.min(speed)
        else:
            features['speed_mean'] = features['speed_std'] = 0
            features['speed_max'] = features['speed_min'] = 0
    else:
        features['speed_mean'] = features['speed_std'] = 0
        features['speed_max'] = features['speed_min'] = 0
    
    # Heart rate features
    if 'heart_rate' in workout and workout['heart_rate'] is not None:
        hr = np.array(workout['heart_rate'], dtype=float)
        hr = hr[~np.isnan(hr)]
        if len(hr) > 0:
            features['hr_mean'] = np.mean(hr)
            features['hr_std'] = np.std(hr)
            features['hr_max'] = np.max(hr)
            features['hr_min'] = np.min(hr)
        else:
            features['hr_mean'] = features['hr_std'] = 0
            features['hr_max'] = features['hr_min'] = 0
    else:
        features['hr_mean'] = features['hr_std'] = 0
        features['hr_max'] = features['hr_min'] = 0
    
    # HR-Speed correlation
    if ('speed' in workout and workout['speed'] is not None and 
        'heart_rate' in workout and workout['heart_rate'] is not None):
        speed_raw = np.array(workout['speed'], dtype=float)
        hr_raw = np.array(workout['heart_rate'], dtype=float)
        valid_mask = ~(np.isnan(speed_raw) | np.isnan(hr_raw))
        speed_valid = speed_raw[valid_mask]
        hr_valid = hr_raw[valid_mask]
        
        if len(speed_valid) > 1 and np.std(speed_valid) > 0 and np.std(hr_valid) > 0:
            corr = np.corrcoef(speed_valid, hr_valid)[0, 1]
            features['hr_speed_corr'] = corr if not np.isnan(corr) else 0
        else:
            features['hr_speed_corr'] = 0
    else:
        features['hr_speed_corr'] = 0
    
    features['speed_variability'] = features['speed_std'] / (features['speed_mean'] + 1e-6)
    
    return features

feature_list = [extract_features(w) for w in workouts]
df_features = pd.DataFrame(feature_list)
df_features['label'] = labels
df_features['sport'] = [w['sport'] for w in workouts]

print(f"   Feature matrix shape: {df_features.shape}")

# 5. Train model
features_to_plot = ['speed_mean', 'speed_std', 'hr_mean', 'hr_std', 'hr_speed_corr', 'speed_variability']
X = df_features[features_to_plot].fillna(0).values
y = df_features['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n5. Training Random Forest...")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

y_pred_test = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n   Test Accuracy: {test_acc:.3f}")

# 6. Evaluation
print("\n6. Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred_test, target_names=['Run', 'Bike']))

# 7. Feature importance
feature_importance = pd.DataFrame({
    'feature': features_to_plot,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n7. Feature Importance:")
print(feature_importance)

# 8. Save results
df_features.to_csv('features_sample.csv', index=False)
print("\n8. Saved features to 'features_sample.csv'")

# 9. Create visualizations
print("\n9. Creating visualizations...")

# Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
colors = ['#2ecc71', '#e74c3c']

for i, feature in enumerate(features_to_plot):
    for label, color, name in [(0, colors[0], 'Run'), (1, colors[1], 'Bike')]:
        data = df_features[df_features['label'] == label][feature]
        axes[i].hist(data, bins=20, alpha=0.6, label=name, color=color)
    axes[i].set_xlabel(feature.replace('_', ' ').title())
    axes[i].set_ylabel('Count')
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
print("   Saved: feature_distributions.png")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=['Run', 'Bike'],
            yticklabels=['Run', 'Bike'],
            cbar_kws={'label': 'Count'})
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("   Saved: confusion_matrix.png")

# Feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='#3498db')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("   Saved: feature_importance.png")

print("\n" + "="*80)
print("COMPLETE EDA FINISHED SUCCESSFULLY!")
print("="*80)
print(f"\nFinal Results:")
print(f"  - Test Accuracy: {test_acc:.1%}")
print(f"  - Total workouts: {len(workouts)}")
print(f"  - Features extracted: {len(features_to_plot)}")
print(f"  - Visualizations saved: 3 PNG files")
print(f"  - Data saved: features_sample.csv")
