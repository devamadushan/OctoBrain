# ************************************* Imports ******************************#

import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from observer import find_truth

#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#

path_brain = 'data/data_brain/brain.hdf5'

#++++++++++++++++++++++++++++++++++* path data in *+++++++++++++++++++++++++#

path_in = 'data/data_in/data_in.csv'

#++++++++++++++++++++++++++++++++++* path data out *+++++++++++++++++++++++++#

path_out = 'data/data_out/data_out.csv'

#++++++++++++++++++++++++++++++++++* path data real *+++++++++++++++++++++++++#

path_true = 'data/data_real/data_real_.csv'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* parametre in  *~~~~~~~~~~~~~~~~~~~~~~~~#

param1 = "Date"
param2 = "Time"
param3 = "VANNE_UTA_CHAUD [I]"
param4 = "VANNE_UTA_FROID [I]"
param5 = "PRESSION_EXTERIEUR [R]"
param6 = "TEMPERATURE_CONSIGNE [R]"
param7 = "TEMPERATURE_ALLER_FROID [R]"
result1 = "TEMPERATURE_REPRISE [R]"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* charger les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

def load_data_brain():
    """Charge le modèle TensorFlow."""
    brain = tf.keras.models.load_model(path_brain)
    return brain

def load_and_reverse_data(data):
    """Inverse les données pour les rendre lisibles."""
    date = data[0]
    time = data[1]
    van_uar_c = data[2]
    van_uar_f = data[3]
    pression_ex = data[4]
    temp_consigne = data[5]
    temp_aller_f = data[6]

    def convert_timestamps_to_dates_np(timestamps):
        """Convertit les timestamps (float32) en dates au format '%d-%m-%Y'."""
        if isinstance(timestamps, np.ndarray):
            timestamps = timestamps.flatten()  # Transforme en tableau 1D

        # Convertir chaque timestamp (numpy.float32) en float natif
        dates = [datetime.fromtimestamp(float(ts)).strftime('%d-%m-%Y') for ts in timestamps]
        return np.array(dates)

    def convert_seconds_to_times_np(seconds):
        """Convertit les secondes en heures (hh:mm:ss)."""
        if isinstance(seconds, np.ndarray):
            seconds = seconds.flatten()
        times = []
        for sec in seconds:
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            times.append(f"{int(h):02}:{int(m):02}:{int(s):02}")
        return np.array(times)

    v1 = convert_timestamps_to_dates_np(date)
    v2 = convert_seconds_to_times_np(time)
    v3 = van_uar_c
    v4 = van_uar_f
    v5 = pression_ex
    v6 = temp_consigne
    v7 = temp_aller_f

    return v1, v2, v3, v4, v5, v6, v7

def load_and_clean_data():
    """Charge et nettoie les données en convertissant les colonnes nécessaires."""
    data_train = pd.read_csv(path_in, sep=';')
    date = data_train['Date'].values
    time = data_train['Time'].values
    van_uar_c = data_train['VANNE_UTA_CHAUD [I]'].values
    van_uar_f = data_train['VANNE_UTA_FROID [I]'].values
    pression_ex = data_train['PRESSION_EXTERIEUR [R]'].values
    temp_consigne = data_train['TEMPERATURE_CONSIGNE [R]'].values
    temp_aller_f = data_train['TEMPERATURE_ALLER_FROID [R]'].values

    # Nettoyer et transformer les données
    def convert_dates_to_timestamps(dates):
        """Convertit les dates au format '%d-%m-%Y' en timestamps."""
        timestamps = []
        for date in dates:
            date_str = date.replace('.', '-')  # Remplace les points par des tirets
            timestamp = datetime.strptime(date_str, '%d-%m-%Y').timestamp()
            timestamps.append(timestamp)
        return np.array(timestamps, dtype=np.float32).reshape(-1, 1)

    def convert_times_to_seconds(times):
        """Convertit les heures au format 'hh:mm:ss' en secondes."""
        seconds = []
        for time in times:
            h, m, s = map(int, time.split(':'))
            total_seconds = h * 3600 + m * 60 + s
            seconds.append(total_seconds)
        return np.array(seconds, dtype=np.float32).reshape(-1, 1)

    def convert_numbers(values):
        """Nettoie les nombres pour les convertir en format flottant."""
        cleaned_values = [float(str(v).replace(',', '.')) for v in values]
        return np.array(cleaned_values, dtype=np.float32).reshape(-1, 1)

    e1 = convert_dates_to_timestamps(date)
    e2 = convert_times_to_seconds(time)
    e3 = convert_numbers(van_uar_c)
    e4 = convert_numbers(van_uar_f)
    e5 = convert_numbers(pression_ex)
    e6 = convert_numbers(temp_consigne)
    e7 = convert_numbers(temp_aller_f)

    return e1, e2, e3, e4, e5, e6, e7

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* enregistrer les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

def save_data_out(tab, out):
    """Sauvegarde les données dans un fichier CSV."""
    reversed_data = load_and_reverse_data(tab)
    v1, v2, v3, v4, v5, v6, v7 = reversed_data
   # print(v1, v2, v3, v4, v5, v6, v7)
    # Créer le DataFrame
    df = pd.DataFrame({
        param1: v1,
        param2: v2,
        param3: v3.flatten() if isinstance(v3, np.ndarray) else v3,
        param4: v4.flatten() if isinstance(v4, np.ndarray) else v4,
        param5: v5.flatten() if isinstance(v5, np.ndarray) else v5,
        param6: v6.flatten() if isinstance(v6, np.ndarray) else v6,
        param7: v7.flatten() if isinstance(v7, np.ndarray) else v7,
        result1: out.flatten() if isinstance(out, np.ndarray) else out
    })

    # Sauvegarder dans un fichier CSV
    df.to_csv(f'{path_out}', index=False)
    print(f"Données sauvegardées dans : {path_out}")
    return True

def observer(x_pred, y_pred):
    """Affiche les prédictions et les valeurs réelles avec les dates en abscisse."""
    # Inverser les données pour récupérer les dates
    reversed_data = load_and_reverse_data(x_pred)
    v1, v2, v3, v4, v5, v6, v7 = reversed_data  # v1 contient les dates

    # Récupérer les valeurs réelles
    y_true = find_truth(x_pred, y_pred)

    # Convertir les dates (v1) en objets datetime pour l'axe X
    dates = [datetime.strptime(date, '%d-%m-%Y') for date in v1]

    # Créer le graphique
    plt.figure(figsize=(15, 8))
    plt.plot(dates, y_true, label="Valeurs réelles ", color="blue")
    plt.plot(dates, y_pred, label="Valeurs prédites", color="red", linestyle="dashed")

    # Ajouter des détails au graphique
    plt.title("Prédictions de la Temperature reprise VS la realité", fontsize=16)
    plt.xlabel("Dates", fontsize=14)
    plt.ylabel("Valeurs de température", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Rotation des labels de l'axe X pour les rendre lisibles
    plt.xticks(rotation=45)

    plt.show()

#:::::::::::::::::::::::::::::::::::*  play *::::::::::::::::::::::::::::::::::#

while True:
    data_in_present = int(input("Si vous avez chargé le fichier CSV dans data_in, entrez 1 : "))
    if data_in_present == 1:
        octopus_brain = load_data_brain()
        x1, x2, x3, x4, x5, x6, x7 = load_and_clean_data()
        prediction = octopus_brain.predict([x1, x2, x3, x4, x5, x6, x7])
        data_out = prediction.flatten()
        entree = [x1, x2, x3, x4, x5, x6, x7]
        observer(entree, data_out)
        save_data_out(entree, data_out)
        data_in_present = 0
    else:
        print("Saisie incorrecte")
