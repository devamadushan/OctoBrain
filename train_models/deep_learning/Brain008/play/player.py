# 22 janvier 2025
# simuler des donnée


# ************************************* Imports ******************************#

import pandas as pd
from keras import models
from keras.src.saving import register_keras_serializable
import tensorflow as tf
import matplotlib.pyplot as plt

# ******************************** Imports Terminal **************************#
#   pip install ...

#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#

path_brain = '../data/brain/brain4.hdf5'

#++++++++++++++++++++++++++++++++++* path data in *+++++++++++++++++++++++++#

path_input = 'data/inputs/inputs.csv'

#++++++++++++++++++++++++++++++++++* path data out *+++++++++++++++++++++++++#

path_out = 'data/output/outputs.csv'

#++++++++++++++++++++++++++++++++++* path data out *+++++++++++++++++++++++++#

path_save = '../data/to_client/data_out.csv'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* parametre in  *~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* charger les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def load_data_brain():
    custom_objects = {"mse": mse}

    brain =  models.load_model(path_brain, custom_objects=custom_objects)
    return brain


def load_data_in():
    x = pd.read_csv(path_input)
    return x


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* enregistrer les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

def save_data_out(entree , sortie):

    inputs = pd.DataFrame(entree)
    output = pd.DataFrame(sortie)
    df = pd.concat([inputs, output], axis=1)
    df.to_csv(path_save, index=False, header=False)


    return True


def observer(x_test , y_pred):

  return True
#:::::::::::::::::::::::::::::::::::*  play *::::::::::::::::::::::::::::::::::#

while True:
    data_in_present = int(input("si vous avez charger le fichier csv dans data_in entrez 1 : "))
    if data_in_present == 1:

        octopus_brain = load_data_brain()
        entree_in = load_data_in()
        print(entree_in)
        prediction = octopus_brain.predict(entree_in)
        data_out = prediction.flatten()
        print(data_out)
        observer(entree_in, data_out)
        save_data_out(entree_in, data_out)
        data_in_present = 0
    else:
        print("saisie incorrect")
