# 29 janvier 2025
# le cerveau du model


# ************************************* Imports ******************************#

from keras import  layers, models,regularizers , optimizers


from trainer import Trainer
import json
import pandas as pd
from tester import Tester
from smartFrame import SmartFrame
from cleaner import Cleaner
# ******************************** Imports Terminal **************************#

#   pip install tenserflow

# **************************** Lire le fishier params Json  *******************#

with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
params = data['params']
result =   data['result']

path_train  = data['path_formater_to_train']['data_to_train']   #data Raw
path_filtered_data_train = data['path_formater_to_train']['data_formated'] #Zone save formated data

path_filtered_data_test = data['path_formater_to_test']['data_formated']
path_test = data['path_formater_to_test']['data_to_test']



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
# -------------------------------------* paths *-----------------------------#

path_inputs_data = f'{data_input}'
path_outputs_data = f'{data_output}'
path_brain = f"{data_brain}"
input_shape = data['input_shape']
output_shape = data['output_shape']

ec = data['echantilloner']
time_echantion =data['time_echantion']

e_train = data['e_data']['e_train']
f_train = data['e_data']['f_train']

epochs = data['epoch_brain']
e_test = data['e_data']['e_test']
f_test = data['e_data']['f_test']
f.close()
# ___________________________________* Brain *_________________________________#




class Brain :

    def __init__(self , name):
        self.name = name

    def create_brain(self,loss, input_dim, hidden_dim, input_shape, output_shape , activation_in, activation_out , kernel , metrics):
        model = models.Sequential()

        # Inputs
        #32
        #relu
        #he_normal
        model.add(layers.Dense(input_dim, input_dim=input_shape, activation=activation_in,
                               kernel_initializer=kernel, kernel_regularizer=regularizers.l2(0.001)
                               ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        # Hidden Layer
        #64
        model.add(layers.Dense(hidden_dim, activation=activation_in,
                               kernel_initializer=kernel, kernel_regularizer=regularizers.l2(0.001)
                               ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        # Output
        #linear

        model.add(layers.Dense(output_shape, activation=activation_out,))
        model.compile(optimizer=optimizers.AdamW(
            learning_rate=0.001, weight_decay=1e-5
        ), loss=loss, metrics=[metrics])

        # mean_squared_error
        #mae
        model.summary()
        return model

    def write_brains_params(self , duree, dim, mae, mse, r2 , echantion , path ):
        with open(f'{path_params_brains}', "r+", encoding="utf-8") as p:
            data_brains = json.load(p)

            if name_brain in data_brains:

                data_brains[name_brain]['params'] = params
                data_brains[name_brain]['outputs'] = result
                data_brains[name_brain]['epochs'] = epochs_brain,
                data_brains[name_brain]['duree'] = f'{duree}',
                data_brains[name_brain]['shape'] = f'{dim}',
                data_brains[name_brain]['MAE'] = round(mae, 2),
                data_brains[name_brain]['MSE'] = round(mse, 2),
                data_brains[name_brain]['R2'] = round(r2, 2),
                data_brains[name_brain]['echantion'] = echantion,
                data_brains[name_brain]['meta'] = path,

            else:
                data_brains[name_brain] = {
                    'params': params,
                    'outputs': result,
                    'epochs': epochs_brain,
                    'duree': f'{duree}',
                    'shape': f'{dim}',
                    'MAE': round(mae, 2),
                    'MSE': round(mse, 2),
                    'R2': round(r2, 2),
                    'ecantion': echantion,
                    'meta': path,

                }
            p.seek(0)
            json.dump(data_brains, p, indent=4, ensure_ascii=False)
            p.truncate()
            p.close()



# ================ Fonction principale ====================
if __name__ == "__main__":
    # Création du modèle
    modele = Brain(name_brain)
    brain = modele.create_brain(
        loss='mean_squared_error',
        input_dim=32,
        hidden_dim=64,
        input_shape=input_shape,
        output_shape=output_shape,
        activation_in='relu',
        activation_out='linear',
        kernel='he_normal',
        metrics='mae'
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
    duree_entrainement , shape = traing.train_modele()
    testing = Tester(path_brain, path_entry_test , path_out_test , brain , epochs, path_simulated_test , path_image)
    mae , mse , r2 ,img = testing.test_model('mae')

    paths_save = [path_entry_test , path_out_test , path_entry_train , path_out_train , formated_data_test , formated_data_train]
    paths_dell = [path_entry_test , path_out_test , path_entry_train , path_out_train , formated_data_test , formated_data_train ,e_train , f_train , e_test , f_test]
    cleaning = Cleaner(paths_save , paths_dell ,path_meta_data , name_brain)
    path = cleaning.compress_and_save()
    modele.write_brains_params(duree_entrainement , shape , mae , mse , r2 , time_echantion , path)

    cleaning.clean_all()