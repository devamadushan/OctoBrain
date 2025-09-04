import numpy as np



import pandas as pd

ecart = pd.Timedelta(minutes=5)

data = pd.read_csv('data/data_raw.csv' , delimiter=';')
print(data[['Date','Time']])
date_time = pd.DataFrame()
date_time['date_time'] = pd.to_datetime(data['Time'] +' '+data['Date'])
#data['date_time'] =  pd.to_datetime(data['Time'] +' '+data['Date'])
data.drop(columns=['Time' ,'Date'], inplace=True)
data.insert(0,'Datetime',date_time['date_time'])
print(data)


def moyenner(tab, data_frame):
    if not tab:  # Vérifier si tab est vide
        print("Le tableau est vide.")
        return

    #print("Nombre d'éléments dans tab:", len(tab))

    cols = data_frame.columns

    # Stocker la première date s'il y a une colonne 'Datetime'
    date_time = tab[0]['Datetime'] if 'Datetime' in tab[0] else None

    # Création du DataFrame à partir de tab
    tempo_tab = pd.DataFrame(tab, columns=cols)

    # Supprimer la colonne 'Datetime' si elle existe
    if 'Datetime' in tempo_tab.columns:
        tempo_tab = tempo_tab.drop(columns=['Datetime'])

    # Convertir toutes les colonnes en float après remplacement des virgules par des points
    def convertir_en_float(colonne):
        return colonne.astype(str).str.replace(',', '.').astype(float)

    try:
        tempo_tab = tempo_tab.apply(convertir_en_float)
    except ValueError as e:
        #print("Erreur de conversion :", e)
        return

    # Calculer la moyenne de chaque colonne
    moyennes = tempo_tab.mean()

    # Construire le DataFrame final
    final_data = pd.DataFrame([moyennes], columns=moyennes.index)

    # Réinsérer la colonne 'Datetime' si elle était présente
    if date_time is not None:
        final_data.insert(0, 'Datetime', date_time)

    # Insérer `final_data` dans `data_frame`
    data_frame = pd.concat([data_frame, final_data], ignore_index=True)

    #print("Nouveau data_frame après insertion :")
    #print(data_frame)

    # Vider la liste tab
    tab.clear()

    return data_frame


test = pd.DataFrame(columns=data.columns)  # DataFrame final
rows_to_add = []

for i in range(0, len(data), 100):
    df = data.iloc[i:i+100]  # Extraction du bloc de 100 lignes
    date_cible = df['Datetime'].iloc[0]  # Initialisation de date_cible

    for index, row in df.iterrows():
        if row['Datetime'] <= date_cible + ecart:
            rows_to_add.append(row)

        if row['Datetime'] >= date_cible + ecart:
            date_cible = row['Datetime']  # Mise à jour de la date cible

            if rows_to_add:
                tempo = moyenner(rows_to_add, test)  # Calcul de la moyenne

                if isinstance(tempo, pd.DataFrame):  # Vérification que tempo est bien un DataFrame
                    test = pd.concat([test, tempo], ignore_index=True)

                rows_to_add = []  # Réinitialisation de rows_to_add

#%%

import pandas as pd

df = pd.read_csv('new_data/data.csv' , sep=';')
print(df.head())
#%%

test = pd.DataFrame(rows_to_add)
test = test.reset_index(drop=True)
print(test)
print(len(test))

#%%
test.to_csv('data/test_2.csv', index=False , sep=';')