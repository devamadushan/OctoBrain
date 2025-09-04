# 21 janvier 2025
# multi parametre simple
from bdb import Breakpoint

# ************************************* Imports ******************************#


import tensorflow as tf
from keras import Model , layers
import numpy as np

# ******************************** Imports Terminal **************************#

#   pip install tenserflow

# ___________________________________* Brain *_________________________________#

## creation
input1 = layers.Input(shape=(1,), name="Input_1")
input2 = layers.Input(shape =(1,), name="Input_2")


x1 = layers.Dense(32, activation="relu")(input1)
x2 = layers.Dense(32, activation="relu")(input2)

combined = layers.concatenate([x1, x2])

x = layers.Dense(64, activation="relu")(combined)

output = layers.Dense(1, activation=None , name="output")(x)
model = Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
model.summary()




# ENtrainement
data1 = np.array([
    [1],  # Exemple 1
    [2],  # Exemple 2
    [3],  # Etc.
    [4]

])

data2 = np.array([
    [1],  # Exemple 1
    [2],  # Exemple 2
    [3],  # Etc.
    [4]


])

# Labels restent les mêmes
labels = np.array([2, 4, 6, 8])

print(f"data 1 {data1}")
# 9. Entraîner le modèle
model.fit([data1, data2], labels, epochs=5000, batch_size=32)







# 10. Faire des prédictions
test_data1 =np.array([
    [1],
    [2],
    [3],
    [4]

])
test_data2 = np.array([
    [1],
    [2],
    [3],
    [4]

])

predictions = model.predict([test_data1, test_data2])
print("Prédictions :", predictions)
