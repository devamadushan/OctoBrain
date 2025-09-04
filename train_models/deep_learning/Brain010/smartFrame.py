from brain import create_brain , write_brains_params
from keras import  layers, models,regularizers , optimizers
from trainer import train
import json
import pandas as pd
from formater import formate_datas
from filter import filter_datas
from tester import test_model ,  test_model_with_others_datas


with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
params = data['params']
result =   data['result']

path_train  = data['path_formater_to_train']['data_to_train']
path_filtered_data_train = data['path_formater_to_train']['data_formated']

data_input = data['path_data_train']['data_input']
data_output = data['path_data_train']['data_output']
data_brain = data['path_brain']
path_params_brains = data['path_param_brains']
name_brain = data['name_brain']
epochs_brain = data['epoch_brain']

formated_data = data['path_formater_to_train']['data_formated']
path_entry = data['path_data_train']['data_input']
path_out = data['path_data_train']['data_output']



path_inputs_data = f'{data_input}' #pour entrainement
print(path_inputs_data)
path_outputs_data = f'{data_output}' #pour entrainement
path_brain = f"{data_brain}"
input_shape = data['input_shape']
output_shape = data['output_shape']



formate_datas(path_train , path_filtered_data_train)

filter_datas(formated_data , path_entry ,path_out)

