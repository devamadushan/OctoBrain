# 19 février 2025
# tester le Model


# ************************************* Imports ******************************#

import pandas as pd
from keras import models
from keras.src.saving import register_keras_serializable
import tensorflow as tf
import json
import  numpy as np
import matplotlib.pyplot as plt
from formater import formate_datas
from filter import filter_datas

# ******************************** Imports Terminal **************************#

#   pip install ...

# **************************** Lire le fishier params Json  *******************#

with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)



path_brain = data['path_brain']

nama_brain = data['name_brain']

sortie = data['result']
params = data['params']
path_input = data['path_formater_to_test']['data_input']

path_out = data['path_formater_to_test']['data_output']

path_save = data['path_formater_to_test']['data_simuled']

path_to_formate = data['path_formater_to_test']['data_to_test']

path_to_save_data_formated = data['path_formater_to_test']['data_formated']



path_of_filtered_data = data['path_formater_to_test']['data_formated']


# -------------------------------------*  *-----------------------------#

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def load_data_brain():
    custom_objects = {"mse": mse}

    brain =  models.load_model(path_brain, custom_objects=custom_objects)
    return brain
def load_data_brain_2(path):
    custom_objects = {"mse": mse}

    brain = models.load_model(path, custom_objects=custom_objects)
    return brain

def load_data_in():
    x = pd.read_csv(path_input)
    return x

def load_data_in_2(path):
    x = pd.read_csv(path)
    return x


def load_waited_data():
    y = pd.read_csv(path_out)
    #print(y)
    return y

def load_waited_data_2(path):
    y = pd.read_csv(path)
    # print(y)
    return y
def load_input_data():
    x = pd.read_csv(path_input)
    return x

def load_input_data_2(path):
    x = pd.read_csv(path)
    return x
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* enregistrer les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

def save_data_out(entree , sortie):

    inputs = pd.DataFrame(entree)
    output = pd.DataFrame(sortie)
    df = pd.concat([inputs, output], axis=1)
    df.to_csv(path_save, index=False, header=False)
    return True

def calcul_loss(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = (y_true - y_pred) ** 2
    loss = np.mean(errors)
    return loss

def observer(y_true , y_pred , input):
    plt.figure(figsize=(10, 6))
    print(y_true)
    #print(y_pred)
    plt.plot( y_true, label="Données attendues", color="green")
    plt.plot( y_pred, label="Données simulées ", color="yellow", linestyle="dashed")
    #plt.plot(input , color="red" , label="Temperature Consigne")
    plt.legend()
    loss = calcul_loss(y_true, y_pred)
    plt.title(f"({nama_brain}) MSE {round(loss,2)}")
    plt.xlabel(f"x {params[0]}")
    plt.ylabel(f"{sortie[0]}")

    plt.show()
    return True
#:::::::::::::::::::::::::::::::::::*  Tester *::::::::::::::::::::::::::::::::::#

def test_model():

        octopus_brain = load_data_brain()
        formate_datas(path_to_formate , path_to_save_data_formated)

        filter_datas(path_of_filtered_data,path_input,path_out )

        entree_in = load_data_in()
        print("entreeeeeee ",entree_in)
        prediction = octopus_brain.predict(entree_in)
        data_out = prediction.flatten()
        print("data_outtt",data_out)
        input_data = load_input_data()
        data_waited = load_waited_data()
        save_data_out(data_waited, data_out)
        moyenne_de_perte = calcul_loss(data_waited, prediction)
        observer(data_waited, prediction , input_data)


#:::::::::::::::::::::::::::::::::::* Other tester *:::::::::::::::::::::::::::::::::#

def test_model_with_others_datas(path_brain_test, path_input_data, path_output_data):
    octopus_brain = load_data_brain_2(path_brain_test)
    #formate_datas(to_formate, save_formated)

    #filter_datas(save_formated, path_input_data, path_output_data)

    entree_in = load_data_in_2(path_input_data)
    #print("entrééee",entree_in)
    prediction = octopus_brain.predict(entree_in)
    #print("predddd ",prediction)
    data_out = prediction.flatten()
    # print(data_out)
    input_data = load_input_data_2(path_input_data)
    data_waited = load_waited_data_2(path_output_data)
    save_data_out(data_waited, data_out)
    moyenne_de_perte = calcul_loss(data_waited, prediction)
    observer(data_waited, prediction , input_data)




#test_model_with_others_datas("brains/brain2.hdf5","train/data/input_train/input.csv " , "train/data/output_train/output.csv")
#test_model()