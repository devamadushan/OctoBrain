
import tensorflow as tf

# Active la gestion mémoire GPU "à la demande"
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

print("GPUs détectés :", gpus)

