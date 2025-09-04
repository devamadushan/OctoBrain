
# ************************************* Imports ******************************#

import pandas as pd
import csv
from dateutil.parser import parse


# ----------------------------* Main programme *-----------------------------#

def add_dateTime(path , delimiter):
    data = pd.read_csv(path, delimiter=delimiter, low_memory=False)

    # Parse une seule fois
    data['datetime'] = pd.to_datetime(
        data['Date'] + ' ' + data['Time'],
        format="%d/%m/%Y %H:%M:%S",
        dayfirst=True,
        errors="coerce"
    )

    # Conversion colonnes numériques
    formated_data = pd.DataFrame()
    for col in data.columns:
        try:
            formated_data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
        except ValueError:
            formated_data[col] = data[col]

    # Mettre datetime en première colonne
    cols = ['datetime'] + [c for c in formated_data.columns if c != 'datetime']
    formated_data = formated_data[cols]

    # Sauvegarde
    formated_data.to_csv(path, sep=delimiter, index=False)



def str_to_datetime(date_str):
    try:
        return parse(date_str)  # Essaie d'analyser la date automatiquement
    except (ValueError, TypeError):
        return None

def echantillonner(path , time, path_save , delimiter) :
    ecart = pd.Timedelta(minutes=time)
    print("echantillon")
    df = pd.read_csv(path , delimiter=delimiter)
    columns = [col.lower() for col in df.columns]
    print(df.head())
    if 'datetime' not in columns:
        print("No datetime")
        add_dateTime(path , delimiter)
    else:
        df['datetime'] = df['datetime'].apply(str_to_datetime)
        cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
        df = df[cols]
        df.to_csv(path, sep=delimiter, index=False)

    with open(path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        # Lecture des colonnes (header)
        cols = next(reader, None)

        # Récupération de la première ligne
        first = next(reader)
        first_date = str_to_datetime(first[0])
        print("Première date :", first_date)

        # Initialisation des dictionnaires
        final_data = {col: [] for col in cols}
        test = {col: [] for col in cols}

        #print("Colonnes détectées :", final_data.keys())

        for row in reader:
            date = str_to_datetime(row[0])
           # print("Date :", date)
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
                final_data['datetime'].append(first_date)
                first_date = date
                test.clear()
                test = {col: [] for col in cols}

    print("<<<<<<<<<<<<<<<<<<< test (Valeurs collectées) >>>>>>>>>>>>>>>>>>>")
   # print(test)

    print("<<<<<<<<<<<<<<<<<<< final_data (Moyennes finales) >>>>>>>>>>>>>>>>>>>")

    pd.DataFrame(final_data).to_csv(path_save, sep=delimiter , index=False)
    print(f"path save :: {path_save}")

#echantillonner("data_raw/to_sample/data_train.csv" , 5 , "data_raw/train/data_train.csv")
#new_data = pd.DataFrame(final_data)
#print(new_data)

#new_data.to_csv('new_data/new_data_melun_1.csv' , sep=';', index=False)