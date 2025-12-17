#!/usr/bin/env python3
"""
Quick test script to verify data loading and basic analysis
"""

import numpy as np
import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("="*60)
print("Running Fraud Detection - Quick Test")
print("="*60)

# Load sample data
print("\n1. Loading data sample (500 workouts)...")
workouts = []
with open('endomondoHR.json', 'r') as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        try:
            workout = ast.literal_eval(line.strip())
            workouts.append(workout)
        except:
            continue

print(f"   Loaded {len(workouts)} workouts")

# Check sports distribution
sports = [w['sport'] for w in workouts]
print("\n2. Sport distribution:")
for sport, count in pd.Series(sports).value_counts().items():
    print(f"   {sport}: {count}")

# Create labels
labels = []
for w in workouts:
    sport = w['sport'].lower()
    if 'run' in sport:
        labels.append(0)  # Genuine
    else:
        labels.append(1)  # Fraud
labels = np.array(labels)

print(f"\n3. Labels created:")
print(f"   Genuine runs: {np.sum(labels == 0)}")
print(f"   Fraudulent (bike): {np.sum(labels == 1)}")

# Extract simple features
print("\n4. Extracting features...")
features = []
valid_labels = []
for i, w in enumerate(workouts):
    # Skip workouts without required fields
    if 'speed' not in w or 'heart_rate' not in w:
        continue
    
    speed = np.array(w['speed'])
    hr = np.array(w['heart_rate'])
    
    # Skip if empty
    if len(speed) == 0 or len(hr) == 0:
        continue
    
    feat = {
        'speed_mean': np.mean(speed),
        'speed_std': np.std(speed),
        'hr_mean': np.mean(hr),
        'hr_std': np.std(hr),
        'hr_speed_corr': np.corrcoef(speed, hr)[0, 1] if len(speed) > 1 else 0,
        'speed_variability': np.std(speed) / (np.mean(speed) + 1e-6)
    }
    features.append(feat)
    valid_labels.append(labels[i])

labels = np.array(valid_labels)

df = pd.DataFrame(features)
print(f"   Feature matrix shape: {df.shape}")

# Train simple model
print("\n5. Training Random Forest classifier...")
X = df.fillna(0).values
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Test Accuracy: {acc:.3f}")

print("\n6. Classification Report:")
if len(np.unique(y_test)) > 1:
    print(classification_report(y_test, y_pred, target_names=['Run', 'Bike']))
else:
    print(f"   Only one class in test set: {np.unique(y_test)}")

print("\n7. Feature Importance:")
feature_names = ['speed_mean', 'speed_std', 'hr_mean', 'hr_std', 'hr_speed_corr', 'speed_variability']
for name, importance in sorted(zip(feature_names, clf.feature_importances_), 
                                key=lambda x: x[1], reverse=True):
    print(f"   {name}: {importance:.3f}")

print("\n" + "="*60)
print("Test completed successfully! ✓")
print("="*60)
print("\nNext step: Open 'EDA_baseline.ipynb' in Jupyter for full analysis")
