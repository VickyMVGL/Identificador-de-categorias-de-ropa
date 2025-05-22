import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math

# descargar el dataset de mnist (Numeros escritos a mano)

datos, metadatos = tfds.load ("mnist", as_supervised=True, with_info=True)

# Obtener en variables separadas los datos de entrenamiento (60k) y de prueba (10k)

datos_entrenamiento, datos_prueba = datos["train"], datos["test"]

# Normalizar los datos de entrenamiento y prueba

def normalizar(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta

datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_prueba = datos_prueba.map(normalizar)

# Agregar datos a cache
datos_entrenamiento = datos_entrenamiento.cache()
datos_prueba = datos_prueba.cache()


#plt.figure(figsize=(10, 10))

# Definir las clases (etiquetas) para los dígitos 0-9

"""
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
"""

#Creando el modelo de red neuronal

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28,1)),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')

])

# Realizando el entrenamiento del modelo
modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

num_ej_entrenamiento = metadatos.splits['train'].num_examples
num_ej_prueba = metadatos.splits['test'].num_examples

tam_lote = 32

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(tam_lote)
datos_prueba = datos_prueba.batch(tam_lote)

historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil( num_ej_entrenamiento / tam_lote))


# Exportar el modelo a un explorador

modelo.save("modelo_mnist.h5")
