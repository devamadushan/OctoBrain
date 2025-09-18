 # Image officielle TF GPU sans Jupyter
FROM tensorflow/tensorflow:2.20.0-gpu

# Qualité de vie & reproductibilité
ENV DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Dépendances Python du projet (hors TF)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# On copie le code si tu veux builder une image “fermée”
# (PyCharm peut aussi monter le volume, voir compose)
COPY . /workspace

# Commande par défaut neutre (on laisse PyCharm piloter le run)
CMD ["bash", "-lc", "echo 'Container prêt pour PyCharm.' && tail -f /dev/null"]
