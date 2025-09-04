# 28 janvier 2025
#permet de formater les donnée



import pandas as pd
from datetime import datetime


# ----------------------------------* Path csv  *----------------------------#

path_train = 'to_trainer/data_train.csv'  #data pour s'entrainer
path_filtered_data_train = '../training/data/filtered_data/filtered_data_train.csv'

path_in = 'to_player/data_in.csv'
path_filtered_data_play = '../play/data/filtered_data/filtered_data_in.csv'
# --------------------------------* nettoyage des données *--------------------#


def clean_data(data):
    try:
        datetime.strptime(data.strip(), "%d.%m.%Y")
        data_return = convert_dates_to_timestamps(data.strip())
    except ValueError:
        try:
            datetime.strptime(data.strip(), "%H:%M:%S")
            data_return = convert_times_to_seconds(data.strip())
        except ValueError:
            data_return = convert_numbers(data.strip())
    return data_return


def convert_dates_to_timestamps(date):
    date_str = date.replace('.', '-')  # Remplace les points par des tirets
    timestamp = datetime.strptime(date_str, '%d-%m-%Y').timestamp()
    return timestamp

def convert_times_to_seconds(time):

    h, m, s = map(int, time.split(':'))
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds

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
        if col == 'Date':
           # print(data['Date'].astype(str).apply(convert_dates_to_timestamps))
            data['Date'] = data['Date'].astype(str).apply(convert_dates_to_timestamps)
            #print(data['Date'])
        if col == 'Time':
            data['Time'] = data['Time'].astype(str).apply(convert_times_to_seconds)

        if col != 'Date' and col != 'Time':
           data[col] = data[col].astype(str).apply(convert_numbers)


    for col in columns:
        filtered_data[col] = data[col]

    return filtered_data

def write_csv(path, datas):
    datas.to_csv(path, index=False)
    return path

#data_train = read_file(path_train)
#train_csv = write_csv(path_filtered_data_train, data_train)


data_play = read_file(path_in)
write_csv(path_filtered_data_play, data_play)