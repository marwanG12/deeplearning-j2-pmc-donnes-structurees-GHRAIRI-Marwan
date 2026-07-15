# Choix : classification MULTICLASS 3 classes (low/medium/high).
# On agrege les qualites 3-8 en 3 groupes. On PERD l'info d'ordre (low<medium<high traites
# comme independants) -> une regression ou ordinal regression la garderait, mais le
# multiclass est le plus simple pour une baseline solide.
# stratify=y OBLIGATOIRE : sans lui, le train pourrait n'avoir aucun exemple "low".
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Miroir stable (UCI a change ses URLs). sep=None + engine='python' = auto-detection du separateur.
wine_url = "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv"
df_wine = pd.read_csv(wine_url, sep=None, engine='python')

print("Distribution des qualites brutes :")
print(df_wine['quality'].value_counts().sort_index())

def map_quality(q):
    if q <= 4:   return 0   # low
    elif q <= 6: return 1   # medium
    else:        return 2   # high

df_wine['quality_3class'] = df_wine['quality'].apply(map_quality)
print("\nDistribution agregee (3 classes) :")
print(df_wine['quality_3class'].value_counts().sort_index())

X_wine = df_wine.drop(['quality', 'quality_3class'], axis=1).values
y_wine = df_wine['quality_3class'].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine)
scaler = StandardScaler()
X_tr_norm = scaler.fit_transform(X_tr)
X_te_norm = scaler.transform(X_te)

n_features = X_tr.shape[1]
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_tr_norm, y_tr, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=0)

best_val_acc = max(history.history['val_accuracy'])
print(f"\nBaseline val_accuracy (max) : {best_val_acc:.4f}")

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_te_norm, verbose=0).argmax(axis=1)
print("\nMatrice de confusion (test) :")
print(confusion_matrix(y_te, y_pred))
print("Classes : 0=low, 1=medium, 2=high")
