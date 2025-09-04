# 29 janvier
# filterer les donner en fonction de la demande du client (params.json)


# ************************************* Imports ******************************#
import json
import pandas as pd



# ******************************** Imports Terminal **************************#
#   pip install ...

# ___________________________________* paths *_________________________________#

filtered_data = 'filtered_data/filtered_data_train.csv'
path_params = '../../params.json'


path_entry = 'inputs/inputs.csv'
path_out= 'output/output.csv'
# -------------------------* Lire les paramétres du client *---------------------#

with open('../../params.json', "r", encoding="utf-8") as f:
    data = json.load(f)
params = data['params']
result =   data['result']

# -------------------------* Lire les paramétres du client *---------------------#

data = pd.read_csv(filtered_data)
entries = data[params]
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

# ------------------------------------*  *-----------------------------#

def write_on_csv(path, d):
    d.to_csv(path,index=False, header=False)


#:::::::::::::::::::::::::::::::::::* Test *:::::::::::::::::::::::::::::::::#
write_on_csv(path_entry,inputs)
write_on_csv(path_out, output)