

# ************************************* Imports ******************************#

import pandas as pd
from keras import models
import matplotlib.pyplot as plt
from tracer import Tracer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ******************************** Imports Terminal **************************#

#   pip install ...


class Tester :
    def __init__(self, path_brain_test, path_input_data, path_output_data , name_brain ,epochs , path_simulated_test , path_image , delimiter):
        self.nama_brain = name_brain
        self.path_brain = path_brain_test
        self.input_data = path_input_data
        self.output_data = path_output_data
        self.simulated_data = None
        self.epochs = epochs
        self.path_simulated_test = path_simulated_test
        self.path_image=path_image
        self.delimiter = delimiter

    def load_data_in(self):
        x = pd.read_csv(self.input_data , sep=self.delimiter)
        print(x)
        return x

    def load_waited_data(self):
        y = pd.read_csv(self.output_data , sep=self.delimiter)
        print(y)
        return y

    def test_model(self,metrics , params , result):
        octopus_brain = self.load_data_brain(metrics)
        entree_in = self.load_data_in()

        prediction = octopus_brain.predict(entree_in)
        data_out = prediction.flatten()
        #print("predictionnnnnnnnnn")
        print(prediction)
        input_data = entree_in
        data_waited = self.load_waited_data()

        mae = mean_absolute_error(data_waited, prediction)
        mse = mean_squared_error(data_waited, prediction)
        r2 = r2_score(data_waited, prediction)
        tracer = Tracer()
        img = tracer.observer_tester(data_waited, prediction, input_data, mae, mse, r2 , params, result)
        self.save_data_out(data_waited, data_out,img)
        return mae, mse, r2

    def load_data_brain(self, metrics):
        custom_objects = {f"{metrics}": metrics}

        brain = models.load_model(self.path_brain, custom_objects=custom_objects)
        return brain



    def save_data_out(self , entree, sortie , img ):
        inputs = pd.DataFrame(entree)
        output = pd.DataFrame(sortie)
        df = pd.concat([inputs, output], axis=1)
        df.to_csv(self.path_simulated_test, sep=self.delimiter, index=False, header=False)
        img.savefig(self.path_image)
        return True

