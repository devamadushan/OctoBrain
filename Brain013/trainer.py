
# ************************************* Imports ******************************#

import pandas as pd
import numpy as np
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import random
import tensorflow as tf
import datetime
from tracer import Tracer

# ******************************** Imports Terminal **************************#
#   pip install ...
#************************************  seed ***********************************#

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
# ----------------------------* Main programme *-----------------------------#


class Trainer:

    def __init__(self , path_brain ,path_inputs_data , path_output_data , brain , epochs ,  delimiter ):
        self.path_brain = path_brain
        self.input_data = path_inputs_data
        self.output_data = path_output_data
        self.brain = brain
        self.epochs = epochs
        self.delimiter = delimiter
    def train_modele(self ):
        x_train, y_train = self.pickups_datas()
        # --- Callbacks anti-surapprentissage ---
        early_stop = EarlyStopping(
            monitor="loss",  # <-- entraînement uniquement
            mode="min",
            patience=3000,
            min_delta=1e-4,
            restore_best_weights=True
        )

        ckpt = ModelCheckpoint(
            filepath=self.path_brain,
            monitor="loss",  # <-- entraînement uniquement
            mode="min",
            save_best_only=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="loss",  # <-- entraînement uniquement
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        callbacks = [early_stop, ckpt, reduce_lr]

        start_time = datetime.datetime.now()
        history = self.brain.fit(
            x_train, y_train,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.2,   # 20% pour validation (indispensable pour détecter l’overfit)
            shuffle=True,
            callbacks=callbacks,
            verbose=2
        )

        # Traces
        tracer = Tracer()
        tracer.observer_trainer(history.history)

        end_time = datetime.datetime.now()
        duree = end_time - start_time
        print(f"départ : {start_time}")
        print(f"fin    : {end_time}")
        print(f"durée  : {duree}")

        # Optionnel : sauvegarde finale (le meilleur est déjà sauvegardé par ModelCheckpoint)
        save_model(self.brain, self.path_brain)

        return duree, str(x_train.shape), history

    def pickups_datas(self):
        x = pd.read_csv(self.input_data , sep=self.delimiter)
        y = pd.read_csv(self.output_data , sep=self.delimiter)

        print("Training Input shape\t: {}".format(x))
        print("Training Output shape\t: {}".format(y))

        return x, y


if __name__ == "__main__":
    import tensorflow as tf

    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
