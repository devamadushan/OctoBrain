
import json
from echantillonneur import echantillonner
from formater import formate_datas


with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
params = data['params']
result =   data['result']

################################################################################
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
#print(path_inputs_data)
path_outputs_data = f'{data_output}' #pour entrainement
path_brain = f"{data_brain}"
input_shape = data['input_shape']
output_shape = data['output_shape']


##################################################################################
ec = data['echantilloner']
time_echantion =data['time_echantion']

e_train = data['e_data']['e_train']
f_train = data['e_data']['f_train']


e_test = data['e_data']['e_test']
f_test = data['e_data']['f_test']

e_play = data['e_data']['e_play']
f_play = data['e_data']['f_play']


def smartf ():
    if ec == True:
        try :
            print("smart frame activated")
            echantillonner(e_train , 5 , f_train)

            echantillonner(e_test, 5, f_test)
            echantillonner(e_play, 5, f_play)
        except Exception as e:
            print(e)


