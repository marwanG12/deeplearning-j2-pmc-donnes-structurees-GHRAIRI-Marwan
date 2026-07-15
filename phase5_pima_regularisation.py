# Objectif : reduire l'ecart train/val (overfitting) via regularisation.
# Ordre teste : baseline (rien) -> L2 seul -> L2 + Dropout. Early Stopping partout.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Preprocessing (identique Phase 4) ---
pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv(pima_url, names=cols)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

def build_pima_regularized(l2_lambda=0.01, use_dropout=False):
    reg = regularizers.l2(l2_lambda) if l2_lambda > 0 else None
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(8,), kernel_regularizer=reg))
    if use_dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=reg))
    if use_dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True)

def run(name, l2_lambda, use_dropout):
    model = build_pima_regularized(l2_lambda, use_dropout)
    h = model.fit(X_train_norm, y_train, epochs=300, validation_split=0.2,
                  callbacks=[early_stopping], verbose=0)
    stop_epoch = len(h.history['val_loss'])
    best_acc = max(h.history['val_accuracy'])
    print(f"{name:18s} | arret epoch {stop_epoch:3d} | max val_acc {best_acc:.4f}")
    return h

print("Config             | Early stop     | Accuracy")
h_base = run("Baseline",  0.0,  False)
h_l2   = run("L2 seul",   0.01, False)
h_drop = run("L2 + Dropout", 0.01, True)

# Courbes val_loss superposees
plt.figure(figsize=(9, 5))
for h, lbl in [(h_base,"Baseline"), (h_l2,"L2 seul"), (h_drop,"L2 + Dropout")]:
    plt.plot(h.history['val_loss'], label=lbl)
plt.xlabel("Epoch"); plt.ylabel("val_loss"); plt.legend()
plt.title("Pima : impact de la regularisation (val_loss)")
plt.savefig("phase5_pima_3configs.png", dpi=100, bbox_inches='tight')
print("\nGraphe sauve : phase5_pima_3configs.png")
