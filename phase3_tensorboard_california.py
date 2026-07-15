import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# --- Preprocessing ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)

def build_regression_model(input_dim=8):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_with_tensorboard(Xtr, ytr, Xva, yva, run_name, epochs=100):
    log_dir = f"logs/fit/{run_name}_" + datetime.datetime.now().strftime("%H%M%S")
    tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = build_regression_model(input_dim=8)
    history = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                        epochs=epochs, batch_size=32, verbose=0, callbacks=[tb])
    print(f"Run '{run_name}' termine. Logs dans {log_dir}")
    return model, history

# Run 1 : normalise (bon comportement)
m_norm, h_norm = train_with_tensorboard(X_train_norm, y_train, X_val_norm, y_val, "california_norm")
# Run 2 : brut (comportement degrade)
m_raw, h_raw = train_with_tensorboard(X_train, y_train, X_val, y_val, "california_raw")

print(f"\nval_loss finale NORM : {h_norm.history['val_loss'][-1]:.4f}")
print(f"val_loss finale RAW  : {h_raw.history['val_loss'][-1]:.4f}")
# Interpretation : le run 'raw' a une val_loss bien plus haute / instable.
# Meme archi, meme optimizer, meme duree : seule la normalisation change.
