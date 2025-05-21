import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
import numpy as np

datos, metadatos = tfds.load('fashion_mnist', with_info=True, as_supervised=True)

datos_entrenamiento, datos_prueba = datos['train'], datos['test']
nombres_clases = metadatos.features['label'].names
print(nombres_clases)

# Normalizar los datos

def normalizar(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32)
    imagen /= 255
    return imagen, etiqueta
# Normalizar los datos de entrenamiento y prueba

datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_prueba = datos_prueba.map(normalizar)

# Agregar datos a cache
datos_entrenamiento = datos_entrenamiento.cache()
datos_prueba = datos_prueba.cache()

# Mostrar imagen 

for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen = imagen.numpy().reshape((28, 28))

#dibujar imagen

"""
plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
"""


"""
plt.figure(figsize=(10, 10))
for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
    imagen = imagen.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.title(nombres_clases[etiqueta])
    plt.axis('off')
plt.show()
"""

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']

)



# Entrenar el modelo

num_ej_entrenamiento = metadatos.splits['train'].num_examples
num_ej_prueba = metadatos.splits['test'].num_examples

tam_lote = 32

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(tam_lote)
datos_prueba = datos_prueba.batch(tam_lote)

historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil( num_ej_entrenamiento / tam_lote))

# Evaluar el modelo
"""
plt.xlabel('Epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss'])
plt.show()
"""

for imagenes_prueba, etiquetas_prueba in datos_prueba.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones = modelo.predict(imagenes_prueba)

def graficar_imagen(i, predicciones_array, etiquetas_reales, imagenes):
    predicciones_array, etiqueta_real, imagen = predicciones_array[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen, cmap=plt.cm.binary)

    etiqueta_prediccion = np.argmax(predicciones_array)
    if etiqueta_prediccion == etiqueta_real:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        nombres_clases[etiqueta_prediccion],
        100*np.max(predicciones_array), 
        nombres_clases[etiqueta_real], 
        color=color
    ))

# Graficar imagenes

def graficar_valor_arreglo(i, predicciones_array, etiquetas_reales):
    predicciones_array, etiqueta_real = predicciones_array[i], etiquetas_reales[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])

    grafica = plt.bar(range(10), predicciones_array, color='#777777')

    plt.ylim([0, 1])

    etiqueta_prediccion = np.argmax(predicciones_array)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')


filas = 5
columnas = 5
num_imagenes = filas * columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
    plt.subplot(filas, 2*columnas, 2*i+1)
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    plt.subplot(filas, 2*columnas, 2*i+2)
    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
    
