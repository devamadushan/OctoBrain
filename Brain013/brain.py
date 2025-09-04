


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
delimiter = data['sep']
f.close()


# ___________________________________* Brain *_________________________________#



class Brain:

    def __init__(self, name):
        self.name = name

    # === Version RÉGRESSION (inchangée) ===
    def create_brain(self, loss, input_dim, hidden_dim, input_shape, output_shape,
                     activation_in, activation_out, kernel, metrics):
        model = models.Sequential(name="regression_brain")

        # Inputs
        model.add(layers.Dense(
            input_dim, input_dim=input_shape, activation=activation_in,
            kernel_initializer=kernel, kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        # Hidden Layer
        model.add(layers.Dense(
            hidden_dim, activation=activation_in,
            kernel_initializer=kernel, kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        # Output
        model.add(layers.Dense(output_shape, activation=activation_out))
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5),
            loss=loss,
            metrics=['mape', 'mse']
        )
        model.summary()
        return model


    def create_brain_supervised_cluster(self, n_classes, input_dim, hidden_dim, input_shape,
                                        activation_in='relu', kernel='he_normal'):

        model = models.Sequential(name="supervised_cluster")

        # Encodeur
        model.add(layers.Dense(
            input_dim, input_dim=input_shape, activation=activation_in,
            kernel_initializer=kernel, kernel_regularizer=regularizers.l2(1e-3)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.15))

        model.add(layers.Dense(
            hidden_dim, activation=activation_in,
            kernel_initializer=kernel, kernel_regularizer=regularizers.l2(1e-3)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.15))

        # Goulot (embeddings) – utile pour visualiser les clusters
        model.add(layers.Dense(max(8, hidden_dim // 2), activation=None, name="latent"))

        # Tête de classification
        model.add(layers.Dense(n_classes, activation='softmax', name='classifier'))

        model.compile(
            optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
            loss='sparse_categorical_crossentropy',  # labels entiers
            metrics=['accuracy']
        )
        model.summary()
        return model

    def write_brains_params(self, duree, dim, mae, mse, r2, ec, echantillon, path, metrics, loss, comment):
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
                data_brains[name_brain]['echantillonner'] = ec,
                data_brains[name_brain]['time_echantillon'] = echantillon,
                data_brains[name_brain]['metrics'] = metrics,
                data_brains[name_brain]['loss'] = loss,
                data_brains[name_brain]['meta'] = path,
                data_brains[name_brain]['comment'] = comment,
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
                    'echantillonner': ec,
                    'time_echantillon': echantillon,
                    'metrics': metrics,
                    'loss': loss,
                    'meta': path,
                    'comment': comment,
                }
            p.seek(0)
            json.dump(data_brains, p, indent=4, ensure_ascii=False)
            p.truncate()
            p.close()

# ================ Fonction principale ====================

if __name__ == "__main__":
    #print(len(result))
    modele = Brain(name_brain)

    #brain = modele.create_brain_supervised_cluster(
    #     n_classes=1,
    #     input_dim=32,
    #     hidden_dim=64,
    #     input_shape=len(params),
    #     activation_in='relu',
    #     kernel='he_normal'
    # )
    # # Méta adaptées côté classification
    # meta_metrics = ["accuracy"]
    # meta_loss = "sparse_categorical_crossentropy"

    brain = modele.create_brain(
        loss=loss,
        input_dim=32,
        hidden_dim=64,
        input_shape=len(params),
        output_shape=len(result),
        activation_in='relu',
        activation_out='linear',
        kernel='he_normal',
        metrics='mae'
    )

    smart = SmartFrame(ec , time_echantion, delimiter)
    smart.training_data(
        e_train=e_train ,
        path_train=path_train ,
        f_train=f_train ,
        path_filtered_data_train = path_filtered_data_train ,
        formated_data = formated_data_train ,
        path_entry = path_entry_train ,
        path_out = path_out_train
    )

    smart.testing_data(e_test=e_test,
                       path_test=path_test ,
                       f_test=f_test ,
                       path_filtered_data_test= path_filtered_data_test,
                       formated_data = formated_data_test,
                       path_entry= path_entry_test,
                       path_out=path_out_test)

    traing = Trainer(path_brain ,path_entry_train , path_out_train , brain , epochs , delimiter)
    duree_entrainement , shape , history = traing.train_modele()
    testing = Tester(path_brain, path_entry_test , path_out_test , name_brain , epochs, path_simulated_test , path_image , delimiter)
    mae , mse , r2  = testing.test_model(metrics , params , result)

    paths_save = [path_entry_test , path_out_test , path_entry_train , path_out_train , formated_data_test , formated_data_train]
    paths_dell = [path_entry_test , path_out_test , path_entry_train , path_out_train , formated_data_test , formated_data_train ,e_train , f_train , e_test , f_test]
    cleaning = Cleaner(paths_save , paths_dell ,path_meta_data , name_brain)
    path = cleaning.compress_and_save()
    modele.write_brains_params(duree_entrainement , shape , mae , mse , r2 , ec,time_echantion , path , metrics , loss , comment)

    #cleaning.clean_all()
