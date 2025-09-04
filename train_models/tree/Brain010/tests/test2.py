import pandas as pd
import random

# Génération d'un dataset simulé pour Quoridor
num_samples = 20

data = {
    "murs_possibles_a_mettre": [random.randint(0, 10) for _ in range(num_samples)],
    "nb_pads_restant_moi": [random.randint(0, 10) for _ in range(num_samples)],
    "nb_pads_restant_adversaire": [random.randint(0, 10) for _ in range(num_samples)],
    "nb_murs": [random.randint(0, 10) for _ in range(num_samples)],
    "position_x_IA": [random.randint(0, 8) for _ in range(num_samples)],
    "position_y_IA": [random.randint(0, 8) for _ in range(num_samples)],
    "position_x_adversaire": [random.randint(0, 8) for _ in range(num_samples)],
    "position_y_adversaire": [random.randint(0, 8) for _ in range(num_samples)]

}

actions = []
for i in range(num_samples):
    if data["murs_possibles_a_mettre"][i] > 5:
        actions.append("bloquer")
    elif data["nb_pads_restant_moi"][i] > data["nb_pads_restant_adversaire"][i]:
        actions.append("avancer")
    else:
        actions.append(random.choice(["avancer", "bloquer"]))

data["action"] = actions

df = pd.DataFrame(data)

# 🔹 Afficher le dataset dans la console
print(df)

# 🔹 Sauvegarder le dataset en fichier CSV
df.to_csv("dataset_quoridor.csv", index=False)

print("✅ Dataset enregistré sous 'dataset_quoridor.csv'")
