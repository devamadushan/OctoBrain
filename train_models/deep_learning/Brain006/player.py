# 22 janvier 2025
# simuler des donnée


# ************************************* Imports ******************************#

import pandas as pd
import tensorflow as  tf
from tensorflow import keras
import matplotlib.pyplot as plt
from data.data_train.generator_train import find_truth
# ******************************** Imports Terminal **************************#
#   pip install ...

#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#

path_brain = 'data/data_brain/brain.hdf5'

#++++++++++++++++++++++++++++++++++* path data in *+++++++++++++++++++++++++#

path_in = 'data/data_in/data_in.csv'

#++++++++++++++++++++++++++++++++++* path data out *+++++++++++++++++++++++++#

path_out = 'data/data_out/data_out.csv'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* parametre in  *~~~~~~~~~~~~~~~~~~~~~~~~#

param = "Temps"
param2 = "Temperature"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* charger les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

def load_data_brain():
    brain = tf.keras.models.load_model(path_brain)
    return brain


def load_data_in():
    data_in = pd.read_csv(path_in)
    data_entry = data_in[f'{param}'].values
    return data_entry


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* enregistrer les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

def save_data_out(entry , out):
    df = pd.DataFrame({f"{param}": entry, f"{param2}": out})
    df.to_csv(f'{path_out}', index=False)
    return True


def observer(x_test , y_pred):
    plt.figure(figsize=(10, 6))
    y_true = find_truth(x_test)
    plt.plot(x_test, y_true, label="Vrai sin(x)", color="blue")
    plt.plot(x_test, y_pred, label="sin(x) prédit", color="red", linestyle="dashed")
    plt.legend()
    plt.title("Prédiction sin")
    plt.xlabel("x (entrée)")
    plt.ylabel("y (sortie) ")
    plt.show()

#:::::::::::::::::::::::::::::::::::*  play *::::::::::::::::::::::::::::::::::#

while True:
    data_in_present = int(input("si vous avez charger le fichier csv dans data_in entrez 1 : "))
    if data_in_present == 1:
        octopus_brain = load_data_brain()
        entree_in = load_data_in()
        #print(entree_in)
        prediction = octopus_brain.predict(entree_in)
        data_out = prediction.flatten()
        #print(data_out)
        observer(entree_in, data_out)
        save_data_out(entree_in, data_out)
        data_in_present = 0
    else:
        print("saisie incorrect")
