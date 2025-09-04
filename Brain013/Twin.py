
# ************************************* Imports ******************************#

from keras import  layers, models,regularizers , optimizers
from trainer import Trainer
import json
from tester import Tester
from smartFrame import SmartFrame
from cleaner import Cleaner
import numpy as np
import random
import tensorflow as tf
from brain import Brain
# ******************************** Imports Terminal **************************#

#   pip install tenserflow

# **************************** Lire le fishier params Json  *******************#

seed =24
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
params = data['params']
result =   data['result']

path_train  = data['path_formater_to_train']['data_to_train']   #data Raw
path_filtered_data_train = data['path_formater_to_train']['data_formated'] #Zone save formated data

path_filtered_data_test = data['path_formater_to_test']['data_formated']
path_test = data['path_formater_to_test']['data_to_test']

metrics = data['metrics']

data_input = data['path_data_train']['data_input']
data_output = data['path_data_train']['data_output']
data_brain = data['path_brain']
path_params_brains = data['path_param_brains']
name_brain = data['name_brain']
epochs_brain = data['epoch_brain']

formated_data_train = data['path_formater_to_train']['data_formated']     #Zone formated data train
formated_data_test = data['path_formater_to_test']['data_formated']

path_entry_train = data['path_data_train']['data_input']
path_out_train = data['path_data_train']['data_output']
path_meta_data = data['meta_data']

path_entry_test =  data['path_formater_to_test']['data_input']
path_out_test =  data['path_formater_to_test']['data_output']
path_simulated_test = data['path_formater_to_test']['data_simulated']
path_image = data['path_formater_to_test']['image']

path_inputs_data = f'{data_input}'
path_outputs_data = f'{data_output}'
path_brain = f"{data_brain}"
#input_shape = data['input_shape']
#output_shape = data['output_shape']

ec = data['echantilloner']
time_echantion =data['time_echantion']

e_train = data['e_data']['e_train']
f_train = data['e_data']['f_train']

epochs = data['epoch_brain']
e_test = data['e_data']['e_test']
f_test = data['e_data']['f_test']

loss = data['loss']
comment = data['comment']
f.close()


if __name__ == "__main__":
    # Création du modèle

    #print(len(result))
    modele = Brain(name_brain)
    brain = modele.create_brain(
        loss=loss,
        input_dim=32,
        hidden_dim=64,
        input_shape=len(params),
        output_shape=len(result),
        activation_in='relu',
        activation_out='linear',
        kernel='he_normal',
        metrics='mae '
    )


    smart = SmartFrame(ec , time_echantion)
    smart.training_data(
        e_train=e_train ,
        path_train=path_train ,
        f_train=f_train ,
        path_filtered_data_train = path_filtered_data_train ,
        formated_data = formated_data_train ,
        path_entry = path_entry_train ,
        path_out = path_out_train)

    smart.testing_data(e_test=e_test,
                       path_test=path_test ,
                       f_test=f_test ,
                       path_filtered_data_test= path_filtered_data_test,
                       formated_data = formated_data_test,
                       path_entry= path_entry_test,
                       path_out=path_out_test)

    traing = Trainer(path_brain ,path_entry_train , path_out_train , brain , epochs)
    duree_entrainement , shape , history = traing.train_modele()
    testing = Tester(path_brain, path_entry_test , path_out_test , name_brain , epochs, path_simulated_test , path_image)
    mae , mse , r2  = testing.test_model(metrics , params , result)

    paths_save = [path_entry_test , path_out_test , path_entry_train , path_out_train , formated_data_test , formated_data_train]
    paths_dell = [path_entry_test , path_out_test , path_entry_train , path_out_train , formated_data_test , formated_data_train ,e_train , f_train , e_test , f_test]
    cleaning = Cleaner(paths_save , paths_dell ,path_meta_data , name_brain)
    path = cleaning.compress_and_save()
    modele.write_brains_params(duree_entrainement , shape , mae , mse , r2 , ec,time_echantion , path , metrics , loss , comment)

    #cleaning.clean_all()
