import numpy as np
import tensorflow as tf
from keras import layers, Model

# Génération des données
x_train = np.linspace(0, 50, 2000).reshape(-1, 1)  # Étendre la plage d'entraînement
y_train = np.sin(x_train % (2 * np.pi))  # Normalisation dans [0, 2π]

# Création du modèle
input_layer = layers.Input(shape=(1,))
x = layers.Dense(128, activation="relu")(input_layer)
x = layers.Dense(128, activation="relu")(x)
output_layer = layers.Dense(1, activation=None)(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
model.summary()

# Entraînement
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1)

# Test
x_test = np.array([[6], [20], [30], [40], [50]])  # Valeurs hors plage initiale
x_test_normalized = x_test % (2 * np.pi)  # Normalisation dans [0, 2π]
y_pred = model.predict(x_test_normalized)

# Résultats
y_true = np.sin(x_test % (2 * np.pi))
print("Entrées (x_test) :", x_test)
print("Prédictions :", y_pred)
print("Vraies valeurs :", y_true)

