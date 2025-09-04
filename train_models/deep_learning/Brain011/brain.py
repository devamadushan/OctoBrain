# 29 janvier 2025
# le cerveau du model


# ************************************* Imports ******************************#

from keras import  layers, models,regularizers , optimizers
from trainer import train
import json
import pandas as pd
from formater import formate_datas
from filter import filter_datas
from tester import test_model
from smartFrame import smartf
# ******************************** Imports Terminal **************************#

#   pip install tenserflow

# **************************** Lire le fishier params Json  *******************#

with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
params = data['params']
result =   data['result']

path_train  = data['path_formater_to_train']['data_to_train']   #data Raw
path_filtered_data_train = data['path_formater_to_train']['data_formated'] #Zone save formated data

data_input = data['path_data_train']['data_input']
data_output = data['path_data_train']['data_output']
data_brain = data['path_brain']
path_params_brains = data['path_param_brains']
name_brain = data['name_brain']
epochs_brain = data['epoch_brain']

formated_data = data['path_formater_to_train']['data_formated']     #Zone formated data train
path_entry = data['path_data_train']['data_input']
path_out = data['path_data_train']['data_output']

# -------------------------------------* paths *-----------------------------#

path_inputs_data = f'{data_input}'
path_outputs_data = f'{data_output}'
path_brain = f"{data_brain}"
input_shape = data['input_shape']
output_shape = data['output_shape']
f.close()
# ___________________________________* Brain *_________________________________#

def get_path_brain():
    return path_brain

def write_brains_params(duree , dim):
    with open(f'{path_params_brains}', "r+", encoding="utf-8") as p:
        data_brains = json.load(p)

        if name_brain in data_brains:

            data_brains[name_brain]['params'] = params
            data_brains[name_brain]['outputs'] = result
            data_brains[name_brain]['epochs'] = epochs_brain,
            data_brains[name_brain]['duree'] = f'{duree}',
            data_brains[name_brain]['shape'] = f'{dim}'

        else:
            data_brains[name_brain] = {
                'params': params,
                'outputs': result,
                'epochs': epochs_brain,
                'duree' : f'{duree}',
                'shape' : f'{dim}'
            }
        p.seek(0)
        json.dump(data_brains, p, indent=4, ensure_ascii=False)
        p.truncate()
        p.close()


def create_brain():
    model = models.Sequential()

    # Inputs
    model.add(layers.Dense(32, input_dim=input_shape, activation='relu',
                           kernel_initializer='he_normal' , kernel_regularizer=regularizers.l2(0.001)
                           ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Hidden Layer
    model.add(layers.Dense(64, activation='relu',
                           kernel_initializer='he_normal' , kernel_regularizer=regularizers.l2(0.001)
                           ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Output
    model.add(layers.Dense(output_shape, activation='linear'))
    model.compile(optimizer=optimizers.AdamW(
        learning_rate=0.001, weight_decay=1e-5
    ), loss='mean_squared_error', metrics=['mae'])
    model.summary()
    return model


def brain():

    brain = create_brain()
    smartf()

    formate_datas(path_train , path_filtered_data_train)    #train

    filter_datas(formated_data , path_entry ,path_out)      #train

    duree_entrainement , shape = train(brain , path_inputs_data, path_outputs_data , path_brain , epochs_brain)

    write_brains_params(duree_entrainement , shape)

    test_model()


brain()