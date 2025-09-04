# 19 février 2025
# tester le Model


# ************************************* Imports ******************************#

import pandas as pd
from keras import models
from keras.src.saving import register_keras_serializable
import tensorflow as tf
import json
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ******************************** Imports Terminal **************************#

#   pip install ...

# **************************** Lire le fishier params Json  *******************#

# with open('params.json', "r", encoding="utf-8") as f:
#     data = json.load(f)
#
#
#
# path_brain = data['path_brain']
#
# nama_brain = data['name_brain']
#
# sortie = data['result']
# params = data['params']
# path_input = data['path_formater_to_test']['data_input']
#
# path_out = data['path_formater_to_test']['data_output']
#
# path_save = data['path_formater_to_test']['data_simuled']
#
# path_to_formate = data['path_formater_to_test']['data_to_test']
#
# path_to_save_data_formated = data['path_formater_to_test']['data_formated']
#
#
#
# path_of_filtered_data = data['path_formater_to_test']['data_formated']


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* enregistrer les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

class Tester :
    def __init__(self, path_brain_test, path_input_data, path_output_data , name_brain ,epochs , path_simulated_test , path_image):
        self.nama_brain = name_brain
        self.path_brain = path_brain_test
        self.input_data = path_input_data
        self.output_data = path_output_data
        self.simulated_data = None
        self.epochs = epochs
        self.path_simulated_test = path_simulated_test
        self.path_image=path_image

    def load_data_in(self):
        x = pd.read_csv(self.input_data)
        return x

    def load_waited_data(self):
        y = pd.read_csv(self.output_data)
        # print(y)
        return y

    def test_model(self,metrics):
        octopus_brain = self.load_data_brain(metrics)
        entree_in = self.load_data_in()

        prediction = octopus_brain.predict(entree_in)
        data_out = prediction.flatten()

        input_data = entree_in
        data_waited = self.load_waited_data()

        mae = mean_absolute_error(data_waited, prediction)
        mse = mean_squared_error(data_waited, prediction)
        r2 = r2_score(data_waited, prediction)
        img = self.observer(data_waited, prediction, input_data, mae, mse, r2 , entree_in.columns, data_waited.columns)
        self.save_data_out(data_waited, data_out,img)
        return mae, mse, r2 , img

    def load_data_brain(self, metrics):
        custom_objects = {f"{metrics}": metrics}

        brain = models.load_model(self.path_brain, custom_objects=custom_objects)
        return brain

    def observer(self, y_true, y_pred, input, mae, mse, r2 , params , sortie):
        plt.figure(figsize=(10, 6))
        print(y_true)
        # print(y_pred)
        plt.plot(y_true, label="Données attendues", color="green")
        plt.plot(y_pred, label="Données simulées ", color="yellow", linestyle="dashed")
        # plt.plot(input , color="red" , label="Temperature Consigne")
        plt.legend()

        plt.title(f"{self.nama_brain} | MAE: {round(mae, 2)} | MSE: {round(mse, 2)} | R²: {round(r2, 2)}")
        plt.xlabel(f"x {params[0]}")
        plt.ylabel(f"{sortie[0]}")

        plt.show()
        return plt

    def save_data_out(self , entree, sortie , img):
        inputs = pd.DataFrame(entree)
        output = pd.DataFrame(sortie)
        df = pd.concat([inputs, output], axis=1)
        df.to_csv(self.path_simulated_test, index=False, header=False)
        img.savefig(self.path_image)
        return True

