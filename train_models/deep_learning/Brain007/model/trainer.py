# trainer.py
import numpy as np
import pandas as pd
from datetime import datetime

# ----------------------------------* Path csv data_train *----------------------------#

file_path = '../data/data_train/data_train_test.csv'  #data pour s'entrainer
path_model = "../data/data_brain/brain.hdf5"  #cerveau du model

epochs = 5000
# ---------------------------* Chargement et nettoyage des données *--------------------#

def load_and_clean_data():
    data_train = pd.read_csv(file_path, sep=';')
    date = data_train['Date'].values
    time = data_train['Time'].values
    van_uar_c = data_train['VANNE_UTA_CHAUD [I]'].values
    van_uar_f = data_train['VANNE_UTA_FROID [I]'].values
    pression_ex = data_train['PRESSION_EXTERIEUR [R]'].values
    temp_consigne = data_train['TEMPERATURE_CONSIGNE [R]'].values
    temp_aller_f = data_train['TEMPERATURE_ALLER_FROID [R]'].values
    temp_reprise = data_train['TEMPERATURE_REPRISE [R]'].values

    # Nettoyer et transformer les données
    def convert_dates_to_timestamps(dates):
        timestamps = []
        for date in dates:
            date_str = date.replace('.', '-')  # Remplace les points par des tirets
            timestamp = datetime.strptime(date_str, '%d-%m-%Y').timestamp()
            timestamps.append(timestamp)
        return np.array(timestamps, dtype=np.float32).reshape(-1, 1)

    def convert_times_to_seconds(times):
        seconds = []
        for time in times:
            h, m, s = map(int, time.split(':'))
            total_seconds = h * 3600 + m * 60 + s
            seconds.append(total_seconds)
        return np.array(seconds, dtype=np.float32).reshape(-1, 1)

    def convert_numbers(values):
        cleaned_values = [float(str(v).replace(',', '.')) for v in values]
        return np.array(cleaned_values, dtype=np.float32).reshape(-1, 1)

    x1 = convert_dates_to_timestamps(date)
    x2 = convert_times_to_seconds(time)
    x3 = convert_numbers(van_uar_c)
    x4 = convert_numbers(van_uar_f)
    x5 = convert_numbers(pression_ex)
    x6 = convert_numbers(temp_consigne)
    x7 = convert_numbers(temp_aller_f)
    s1 = convert_numbers(temp_reprise)

    return x1, x2, x3, x4, x5, x6, x7, s1

# ----------------------------------* Entraîner le modèle *-----------------------------------#

def train_brain(brain):

    x1, x2, x3, x4, x5, x6, x7, s1 = load_and_clean_data()

    # Début de l'entraînement
    start_time = datetime.now()
    print(f"Début de l'entraînement : {start_time}")

    inputs = [x1, x2, x3, x4, x5, x6, x7]

    # Entraîner le modèle
    brain.fit(inputs, s1, epochs=epochs, batch_size=64)

    # Fin de l'entraînement
    end_time = datetime.now()
    print(f"Fin de l'entraînement : {end_time}")
    print(f"Durée totale : {end_time - start_time}")

    # Sauvegarde du modèle

    brain.save(path_model)
    print(f"Modèle sauvegardé à : {path_model}")
    return path_model
