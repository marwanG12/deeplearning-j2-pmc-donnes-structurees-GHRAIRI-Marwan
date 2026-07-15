# Pima est desequilibre (~65% classe 0 / 35% classe 1).
# Un modele qui predirait TOUJOURS 0 afficherait 65% d'accuracy sans rien apprendre.
# -> on verifie la distribution AVANT de juger, et on check model.predict().mean() ~ 0.35.
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)

# Distribution des classes
print("Distribution des classes :")
print(df['Outcome'].value_counts())
majority_acc = df['Outcome'].value_counts().max() / len(df)
print(f"-> un modele qui predit toujours la majorite ferait {majority_acc:.1%} d'accuracy\n")

# Zeros suspects (NaN deguises : physiologiquement impossibles)
print("Colonnes avec des zeros suspects :")
print((df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] == 0).sum())

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)   # fit sur train uniquement
X_test_norm = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(8,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_norm, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=0)

best_val_acc = max(history.history['val_accuracy'])
pred_mean = model.predict(X_test_norm, verbose=0).mean()
print(f"\nBaseline val_accuracy (max) : {best_val_acc:.4f}")
print(f"model.predict().mean() : {pred_mean:.3f} (doit etre proche de 0.35, pas 0.05)")
