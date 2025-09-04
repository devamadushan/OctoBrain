# 28 janvier 2025
#permet de formater les donnée



import pandas as pd
from datetime import datetime
import json


# **************************** Lire le fishier params Json  *******************#

with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)

path_train  = data['path_formater_to_train']['data_to_train']
path_filtered_data_train = data['path_formater_to_train']['data_formated']


# ----------------------------------* Path csv  *----------------------------#

#path_train = 'data/to_trainer/data_train.csv'  #data pour s'entrainer
#path_filtered_data_train = 'training/data/formated_data/formated_data.csv'

#path_in = 'data/to_player/data_play.csv'
#path_filtered_data_play = 'play/data/filtered_data/filtered_data_in.csv'

# --------------------------------* nettoyage des données *--------------------#




def convert_dates_to_timestamps(date):
    date_str = date.replace('.', '-')  # Remplace les points par des tirets
    #print(date_str)
    timestamp = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp()
    return timestamp


def convert_numbers(value):
    try:
        # Vérifie si la valeur est vide ou None
        if not value or str(value).strip() == "":
            val = 0
            return val

        # Convertit la valeur en float
        cleaned_values = [float(str(value).replace(',', '.'))]
        return str(cleaned_values[0])

    except ValueError:
        # En cas d'erreur de conversion, retourne None
        print(f"Impossible de convertir : '{value}'")
        return None



# --------------------------------* Chargement  des données *------------------------#



def read_file(path):

    data = pd.read_csv(path , sep=';')
    columns = data.columns
    filtered_data = pd.DataFrame(columns = columns)

    for col in columns:
        #print(col)
        if col == 'Datetime':
            #print(data['Date'].astype(str).apply(convert_dates_to_timestamps))
            data['Datetime'] = data['Datetime'].astype(str).apply(convert_dates_to_timestamps)
           # print(data['Datetime'])

        if col != 'Datetime':
           data[col] = data[col].astype(str).apply(convert_numbers)


    for col in columns:
        filtered_data[col] = data[col]

    return filtered_data

def write_csv(path, datas):
    df = pd.DataFrame(datas)
    df.to_csv(path, index=False)
    #print(datas)
    return path

def formate_datas(path_data_to_formate , path_filtered_data):
    data_train = read_file(path_data_to_formate)
    write_csv(path_filtered_data, data_train)

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  data formate success !!! >>>>>>>>>>>>>>>>>>>>>>>>>>>")

#formate_datas('../datas/E4_2/C4/C4_main.csv' , '../datas/E4_2/C4/C4_main_formated.csv')

