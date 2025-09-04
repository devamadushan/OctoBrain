from datetime import datetime

def extraire_dates(data, date_format='%d-%m-%y'):
    """
    Extrait uniquement les éléments qui correspondent au format de date donné.
    """
    dates = []
    for element in data:
        try:
            # Tenter de convertir en date
            date = datetime.strptime(str(element), date_format)
            dates.append(date)
        except ValueError:
            continue  # Ignorer les éléments non conformes
    return dates

# Exemple d'utilisation
data = ["15-01-23", 42, 3.14, "16-02-23", "hello", "25-12-22", "10:30"]
dates = extraire_dates(data)
print(dates)
