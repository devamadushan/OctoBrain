# 22 janvier 2025
# simuler des donnée


# ************************************* Imports ******************************#

import pandas as pd
from keras import models
from keras.src.saving import register_keras_serializable
import tensorflow as tf
from cleaner import Cleaner
import json

from tensorflow.python.keras.saving.saved_model.save_impl import metrics
from tracer import Tracer
from smartFrame import SmartFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# ******************************** Imports Terminal **************************#

#   pip install ...

model = "brain_1"
path_brain = f'brains/{model}.keras'
# **************************** Lire le fishier params Json  *******************#

with open('brains/brains_params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
    if model in data:
        params = data[model]['params']
        result = data[model]['outputs']
        mse = data[model]['MSE']
        mae = data[model]['MAE']
        r2 = data[model]['R2']
        metrics = data[model]['metrics']
        ech = data[model]['echantillonner']
        time_ech = data[model]['time_echantillon']
        metrics = data[model]['metrics']
        meta = data[model]['meta']

with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
    path_input = data['path_data_play']['data_input']   # apres filter
    path_save = data['path_data_play']['data_output']   #apres la prédiction
    data_to_play  = data['path_formater_to_play']['data_to_play']  # apres le formatage
    path_of_formated_data_play  = data['path_formater_to_play']['data_formated']
    e_play = data['e_data']['e_play']
    f_play = data['e_data']['f_play']
    path_output = data['path_formater_to_play']['data_output']
    result = data['path_formater_to_play']['result']
    delimiter = data['sep']

ech = True
time_ech = 5
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* enregistrer les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#


#:::::::::::::::::::::::::::::::::::*  play *::::::::::::::::::::::::::::::::::#



class Player:
    def __init__(self , paths_brain , paths_input , paths_save , metrics ,paths_output , delimiter):
        self.paths_brain = paths_brain
        self.input = paths_input
        self.save_paths = paths_save
        self.metrics = metrics
        self.output = paths_output
        self.delimiter = delimiter


    def play(self, result):
        if result != True :
            try:
                octopus_brain = self.load_data_brain()
                entree_in = self.load_data_in()
                #print(entree_in)
                prediction = octopus_brain.predict(entree_in)
                data_out = prediction.flatten()
                # print(data_out)

                self.save_data_out(entree_in, prediction)
                #observer(entree_in, prediction)
                return data_out
            except Exception as e:
                print(e)

        else :
            try:
                octopus_brain = self.load_data_brain()
                entree_in = self.load_data_in()
                waited_out = self.load_data_out()
                # print(entree_in)
                prediction = octopus_brain.predict(entree_in)
                data_predict = prediction.flatten()
                # print(data_out)
                self.save_data_out(entree_in, prediction)
                # observer(entree_in, prediction)

                mae = mean_absolute_error(waited_out, prediction)
                mse = mean_squared_error(waited_out, prediction)
                r2 = r2_score(waited_out, prediction)
                tracer = Tracer()
                print("oooo")
                tracer.observer_tester(waited_out,data_predict,entree_in ,mae,mse , r2 , params ,result)
                return data_predict
            except Exception as e:
                print(e)

    def load_data_in(self):
        x = pd.read_csv(self.input , sep = self.delimiter)
        return x

    def load_data_out(self):
        x = pd.read_csv(self.output , sep=self.delimiter)
        return x

    def load_data_brain(self):
        custom_objects = {f"{self.metrics}": self.metrics}

        brain = models.load_model(path_brain, custom_objects=custom_objects)
        return brain

    def save_data_out(self,entree, sortie):
        inputs = pd.DataFrame(entree)
        output = pd.DataFrame(sortie)
        df = pd.concat([inputs, output], axis=1)
        df.to_csv(self.save_paths, index=False, header=False , sep=self.delimiter)
        return True

    @register_keras_serializable()
    def mse(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))


if __name__ == "__main__":
    playing = Player(
        paths_brain=path_brain ,
        paths_input=path_input,
        paths_save=path_save,
        paths_output = path_output,
        metrics=metrics,
        delimiter=delimiter
    )
    smart = SmartFrame(ech=ech,
                       time_echantion=time_ech , delimiter=delimiter)

    if result :

        smart.testing_data(e_test=e_play,
                           f_test=f_play,
                           path_test=f_play,
                           formated_data=path_of_formated_data_play,
                           path_filtered_data_test = path_of_formated_data_play,
                           path_entry=path_input,
                           path_out=path_output)
    else:
        smart.playing_data(e_play=e_play,
                           f_play= f_play ,
                           data_to_play=f_play,
                           path_of_formated_data_play=path_of_formated_data_play ,
                           path_input_play = path_input )

    playing.play(result)

    path_save = [path_save]
    path_dell = [e_play , f_play , path_of_formated_data_play , path_input]

    cleaning = Cleaner( path_save , path_dell , meta ,model)
    #cleaning.clean_all()

