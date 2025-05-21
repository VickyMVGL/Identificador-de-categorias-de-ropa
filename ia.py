import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math

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

plt.xlabel('Epochs')
plt.ylabel('Magnitud de perdida')