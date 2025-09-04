import math


# Fonction pour calculer l'entropie d'un ensemble de données
def entropy(data):
    total = len(data)

    counts = {}

    # Compter combien de "Oui" et combien de "Non"
    for label in data:
        counts[label] = counts.get(label, 0) + 1

    # Calculer l'entropie
    ent = 0
    for count in counts.values():
        print(f'count daa {count}')
        p = count / total
        print(f'ppppp {p}')
        ent -= p * math.log2(p)  # Formule de Shannon
    print(ent)
    print(counts)
    return ent


# Notre ensemble de données (Fait du sport ? Oui / Non)
dataset = [
    ("Alice", 20, "Oui"),
    ("Bob", 25, "Oui"),
    ("sheinez", 25, "Oui"),
    ("Nouria", 30, "Non"),
    ("Charlie", 30, "Non"),
    ("David", 35, "Non"),
    ("Emma", 40, "Non")
]

# Séparer en deux groupes : âge < 30 et âge >= 30
groupe1 = [label for _, age, label in dataset if age < 30]  # Avant 30 ans
groupe2 = [label for _, age, label in dataset if age >= 30]  # 30 ans et plus

# Calcul de l'entropie avant séparation
entropie_initiale = entropy([label for _, _, label in dataset])

# Calcul de l'entropie après séparation
entropie_g1 = entropy(groupe1)
entropie_g2 = entropy(groupe2)

# Calcul du gain d'information
p1 = len(groupe1) / len(dataset)
p2 = len(groupe2) / len(dataset)
gain_info = entropie_initiale - (p1 * entropie_g1 + p2 * entropie_g2)

# Affichage des résultats
print(f"Entropie initiale : {entropie_initiale:.3f}")
print(f"Entropie après séparation (âge < 30) : {entropie_g1:.3f}")
print(f"Entropie après séparation (âge >= 30) : {entropie_g2:.3f}")
print(f"Gain d'information : {gain_info:.3f}")
