# Ordre choisi : SPLIT d'abord, PUIS scaler.fit(X_train) uniquement.
# Pourquoi : fitter le scaler sur X entier ferait "voir" les stats du test set
# au preprocessing (data leakage) -> scores trop optimistes. On fit sur le train seul.
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y = housing.data, housing.target

# 1er split : train+val (80%) / test (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2e split : train (80% du trainval) / val (20% du trainval)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42)

# Normalisation : fit sur X_train UNIQUEMENT
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

print(f"X_train shape : {X_train_norm.shape}")
print(f"X_val shape   : {X_val_norm.shape}")
print(f"X_test shape  : {X_test_norm.shape}")
print(f"\nX_train_norm mean (par feature) :\n{X_train_norm.mean(axis=0).round(3)}")
print(f"\nX_train_norm std (par feature)  :\n{X_train_norm.std(axis=0).round(3)}")
print(f"\nFeature names ({len(housing.feature_names)}) : {housing.feature_names}")
assert len(housing.feature_names) == 8, "Il devrait y avoir 8 features"
print("\nOK : 8 features, normalisation propre.")
