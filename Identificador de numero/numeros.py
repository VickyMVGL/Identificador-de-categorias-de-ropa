import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# descargar el dataset de mnist (Numeros escritos a mano)

datos, metadados = tfds.load ("mnist", as_supervised=True, with_info=True)

# Obtener en variables separadas los datos de entrenamiento (60k) y de prueba (10k)

datos_entrenamiento, datos_prueba = datos["train"], datos["test"]

# Normalizar los datos de entrenamiento y prueba

def normalizar(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta

datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_prueba = datos_prueba.map(normalizar)

plt.figure(figsize=(10, 10))

# Definir las clases (etiquetas) para los dígitos 0-9
clases = [str(i) for i in range(10)]

for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
    imagen = imagen.numpy().reshape(28, 28)
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(clases[int(etiqueta)])
plt.show()