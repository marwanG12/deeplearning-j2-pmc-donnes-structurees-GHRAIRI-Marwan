# Sortie régression = Dense(1) SANS activation.
# Une variante avec activation='sigmoid' tournerait sans erreur mais serait INUTILISABLE :
# sigmoid écrase la sortie entre 0 et 1, alors que les prix vont de ~0.15 à 5.0
# (centaines de milliers de $). Le modèle ne pourrait jamais prédire > 1.
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# --- Preprocessing (identique Phase 1) ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# --- Modèle ---
def build_regression_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)                      # régression : pas d'activation
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_regression_model(input_dim=8)
model.summary()

history = model.fit(X_train_norm, y_train, epochs=100, batch_size=32,
                    validation_data=(X_val_norm, y_val), verbose=1)

test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\nMAE test : {test_mae:.4f} (en centaines de milliers de $)")
print(f"MSE test : {test_loss:.4f}")
