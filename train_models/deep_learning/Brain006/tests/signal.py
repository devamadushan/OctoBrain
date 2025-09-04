# 21 janvier 2025
#


# ************************************* Imports ******************************#


import tensorflow as tf
from keras import Model , layers , callbacks
import numpy as np
import matplotlib.pyplot as plt

# ******************************** Imports Terminal **************************#

#   pip install tenserflow

# ___________________________________* Brain *_________________________________#

## creation
input1 = layers.Input(shape=(1,), name="Input_1")



x1 = layers.Dense(64, activation="relu")(input1) ## in
x = layers.Dense(64, activation="relu")(x1) #hyden
output = layers.Dense(1, activation=None , name="output")(x) #out


model = Model(inputs=input1, outputs=output)
model.compile(optimizer="adamW", loss="mean_squared_error", metrics=["mae"])
model.summary()


x_train = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # Angles pour test
y_train = x_train *2 #


model.fit(x_train, y_train, epochs=5000, batch_size=64)




# Test
x_test = np.linspace(0, 2 * np.pi, 300).reshape(-1, 1)  # Angles pour test
y_true = x_test *2 # Valeurs réelles
y_pred = model.predict(x_test)  # Valeurs prédites

for i in range(1,200):
    print(f"entry {x_test[i]}")
    print(f"true {y_true[i]}")
    print(f" prediction {y_pred[i]}")




# Afficher résultats

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_true, label="Vrai sin(x)", color="blue")
plt.plot(x_test, y_pred, label="sin(x) prédit", color="red", linestyle="dashed")
plt.legend()
plt.title("Prédiction de sin(x) avec Keras")
plt.xlabel("x (entrée)")
plt.ylabel("y (sortie) ")
plt.show()

while True:
    x = int(float(input('entry :')))
    prediction = model.predict(np.array([x]).reshape(-1,1))
    print(f"entry {np.array([x]).reshape(-1,1)}")
    print(f"True {x*2}")
    print(f'Sin  :{prediction} ')