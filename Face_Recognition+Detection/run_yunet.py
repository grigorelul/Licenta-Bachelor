import tensorflow as tf
from tensorflow.python.client import device_lib

# Setarea nivelului de logare la DEBUG
import logging
tf.get_logger().setLevel(logging.DEBUG)

# Verificarea versiunii TensorFlow
print("TensorFlow Version:", tf.__version__)

# Listarea dispozitivelor disponibile
devices = device_lib.list_local_devices()
for device in devices:
    print(device)

# Setarea configurării GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )
        print(f"GPU configurat: {physical_devices[0]}")
    except RuntimeError as e:
        print(e)

print("Număr de GPU-uri disponibile: ", len(tf.config.experimental.list_physical_devices('GPU')))
