
# ************************************* Imports ******************************#

import os
import zipfile
from datetime import datetime
import uuid

# ----------------------------* Main programme *-----------------------------#

class Cleaner:
    def __init__(self, paths_save , paths_dell , meta_data_path , name_brain):
        self.paths_save = paths_save
        self.paths_dell = paths_dell
        self.meta_data_path = meta_data_path
        self.name_brain = name_brain
    def compress_and_save(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        zip_filename = f"{self.name_brain}_{timestamp}_{unique_id}.zip"
        zip_path = os.path.join(self.meta_data_path, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.paths_save:
                if os.path.isfile(file_path):
                    try:
                        zipf.write(file_path, arcname=os.path.basename(file_path))
                        print(f" Ajouté dans l'archive : {file_path}")
                    except Exception as e:
                        print(f" Erreur pour {file_path} : {e}")
                else:
                    print(f" Fichier introuvable : {file_path}")

        print(f"\n Archive créée ici : {zip_path}")
        return zip_path

    def clean_all(self):
        for path in self.paths_dell:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    print(f" Aucun fichier à supprimer ou chemin invalide : {path}")
            except Exception as e:
                print(f" Erreur lors de la suppression de {path} : {e}")

