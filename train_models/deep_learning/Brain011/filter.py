# 29 janvier
# filterer les donner en fonction de la demande du client (params.json)


# ************************************* Imports ******************************#
import json
import pandas as pd



# ******************************** Imports Terminal **************************#
#   pip install ...

# ___________________________________* paths *_________________________________#

#filtered_data = 'filtered_data/filtered_data_train.csv'
path_params = 'params.json'


#path_entry = 'training/data/inputs/inputs.csv'
#path_out= 'training/data/output/output.csv'
# -------------------------* Lire les paramétres du client *---------------------#

with open(f'{path_params}', "r", encoding="utf-8") as f:
    data = json.load(f)
formated_data = data['path_formater_to_train']['data_formated']
path_entry = data['path_data_train']['data_input']
path_out = data['path_data_train']['data_output']
params = data['params']
result =   data['result']

# -------------------------* Lire les paramétres du client *---------------------#
def get_formated_data(formated):
    data = pd.read_csv(formated )
    entries = data[params]
    print("formater : ",entries)
    print(" formater ouuuuutttttt", data[result])
    out = data[result]

    entree = []
    sortie = []

    for row in entries.values:
        entree.append(row)


    for row in out.values:
        sortie.append(row)
        #print(row)

    inputs = pd.DataFrame(entree)
    output = pd.DataFrame(sortie)
    return inputs, output
# ------------------------------------*  *-----------------------------#

def write_on_csv(path, d):
    d = pd.DataFrame(d)
    d = d.fillna(0)
    d.to_csv(path,index=False, header=False)


def filter_datas(f , entry , out):
    print(f)
    print(entry)
    inputs, output = get_formated_data(f)
    print(inputs)
    write_on_csv(entry,inputs)
    write_on_csv(out, output)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  data filter to train success !!! >>>>>>>>>>>>>>>>>>>>>>>>>>>")

def filter_datas_to_player(f , entry ):
    print(f)
    print(entry)
    inputs, output = get_formated_data(f)
    print(inputs)
    write_on_csv(entry,inputs)

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  data filter to play success !!! >>>>>>>>>>>>>>>>>>>>>>>>>>>")

#filter_datas(formated_data , path_entry, path_out)