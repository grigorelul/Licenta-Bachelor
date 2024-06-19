import os
import tempfile

# Setează variabilele de mediu pentru TEMP și TMP
tempdir = 'D:/TEMP'
os.environ['TEMP'] = tempdir
os.environ['TMP'] = tempdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce nivelul de logare
os.environ['TF_TEMP'] = tempdir
os.environ['TF_CACHE_DIR'] = tempdir

# Setează directoarele temporare pentru modulul tempfile
tempfile.tempdir = tempdir

import tensorflow as tf

# Verifică dacă există GPU și configurează memoria corespunzător
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU found and memory growth set: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, running on CPU")

# Verifică dacă directoarele temporare sunt setate corect
print(f"Python TEMP directory: {tempfile.gettempdir()}")
print(f"TensorFlow TEMP directory: {os.getenv('TF_TEMP')}")
print(f"TensorFlow CACHE directory: {os.getenv('TF_CACHE_DIR')}")