import tensorflow as tf

print("Număr de GPU-uri disponibile: ", len(tf.config.experimental.list_physical_devices('GPU')))
