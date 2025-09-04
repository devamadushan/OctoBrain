# 22 janvier 2025
# simuler des donnée


# ************************************* Imports ******************************#

import pandas as pd
from keras import models
from keras.src.saving import register_keras_serializable
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from formater import formate_datas
from filter import filter_datas_to_player


# ******************************** Imports Terminal **************************#

#   pip install ...

model = "brain2"
path_brain = f'brains/{model}.hdf5'
# **************************** Lire le fishier params Json  *******************#

with open('brains/brains_params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
    if model in data:
        params = data[model]['params']
        result = data[model]['outputs']

with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
    path_input = data['path_data_play']['data_input']
    path_save = data['path_data_play']['data_output']
    data_to_play  = data['path_formater_to_play']['data_to_play']
    path_of_formated_data_play  = data['path_formater_to_play']['data_formated']


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


def observer(entre , prediction):
    print(entre.head())
    return True
#:::::::::::::::::::::::::::::::::::*  play *::::::::::::::::::::::::::::::::::#

def play():
    octopus_brain = load_data_brain()
    entree_in = load_data_in()
    print(entree_in)
    prediction = octopus_brain.predict(entree_in)
    data_out = prediction.flatten()
    # print(data_out)

    save_data_out(entree_in, prediction)
    observer(entree_in, prediction)



formate_datas(data_to_play ,path_of_formated_data_play)
filter_datas_to_player(path_of_formated_data_play , path_input)
#formate_datas(path_of_formated_data_play ,path_input )
play()