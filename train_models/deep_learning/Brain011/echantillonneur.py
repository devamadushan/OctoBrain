from os.path import split
from xmlrpc.client import DateTime
import numpy as np
import pandas as pd
import csv
import os
from dateutil.parser import parse

print(os.getcwd())

def add_dateTime(path) :
    data = pd.read_csv(path , delimiter=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    #print(data[['Date','Time']])
    date_time = pd.DataFrame()
    date_time['date_time'] = pd.to_datetime(data['Time'] +' '+data['Date'])
    data['date_time'] =  pd.to_datetime(data['Time'] +' '+data['Date'])
    data.drop(columns=['date_time'], inplace=True)
    data.insert(0,'Datetime',date_time['date_time'])

    formated_data = pd.DataFrame()
    for col in data.columns:
        try:
            formated_data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
        except ValueError as e:
            formated_data[col] = data[col]
            print("Erreur de conversion :", e)


    formated_data.to_csv(path , sep=';', index=False)



#add_dateTime('Brain010/data_raw/to_sample/data_train.csv')


def str_to_datetime(date_str):
    try:
        return parse(date_str)  # Essaie d'analyser la date automatiquement
    except (ValueError, TypeError):
        return None

def echantillonner(path , time, path_save) :
    ecart = pd.Timedelta(minutes=time)
    print("echantion")
    df = pd.read_csv(path , delimiter=';')
    if 'Datetime' not in df.columns:
        add_dateTime(path)
    with open(path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')

        # Lecture des colonnes (header)
        cols = next(reader, None)

        # Récupération de la première ligne
        first = next(reader)
        first_date = str_to_datetime(first[0])
        #print("Première date :", first_date)

        # Initialisation des dictionnaires
        final_data = {col: [] for col in cols}
        test = {col: [] for col in cols}

        #print("Colonnes détectées :", final_data.keys())

        for row in reader:
            date = str_to_datetime(row[0])
            #print("Date :", date)
            count = 0
            if date <= first_date + ecart:
                count += 1
                for index, valeur in enumerate(row):
                    # Convertir les valeurs en float si possible
                    try:
                        valeur = float(valeur) if valeur.replace('.', '', 1).replace('-', '', 1).isdigit() else valeur
                    except ValueError:
                        pass  # Garde la valeur en tant que chaîne si elle ne peut pas être convertie

                    test[cols[index]].append(valeur)

                #print(f"Ligne ajoutée (index {count}):", test)

            if date >= first_date + ecart:


                # Calculer la moyenne des colonnes numériques
                for col in cols[1:]:  # On saute la colonne "Datetime"
                    try:
                        # Convertir les valeurs en float et ignorer les erreurs

                        valeurs = [float(x) for x in test[col] if isinstance(x, (int, float))]
                        #print(valeurs)
                        moyenne = sum(valeurs) / len(valeurs) if valeurs else 0
                        #print(moyenne)
                        # Vérifier que les moyennes sont bien stockées
                        final_data[col].append(moyenne)

                          # Mise à jour de la première date pour le prochain bloc
                    except ValueError:
                        print(f"Impossible de calculer la moyenne pour {col} (valeurs non numériques)")
                final_data['Datetime'].append(first_date)
                first_date = date
                test.clear()
                test = {col: [] for col in cols}

    print("<<<<<<<<<<<<<<<<<<< test (Valeurs collectées) >>>>>>>>>>>>>>>>>>>")
    #print(test)

    print("<<<<<<<<<<<<<<<<<<< final_data (Moyennes finales) >>>>>>>>>>>>>>>>>>>")
    #print(final_data)
    pd.DataFrame(final_data).to_csv(path_save, sep=';' , index=False)


#echantillonner("data_raw/to_sample/data_train.csv" , 5 , "data_raw/train/data_train.csv")
#new_data = pd.DataFrame(final_data)
#print(new_data)

#new_data.to_csv('new_data/new_data_melun_1.csv' , sep=';', index=False)