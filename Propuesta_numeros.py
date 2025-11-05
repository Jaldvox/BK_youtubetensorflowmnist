from __future__ import absolute_import, division, print_function, unicode_literals

# Importaciones esenciales
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime # Para nombrar los directorios de logs

# Configuración del logger para evitar mensajes de advertencia excesivos
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print("Versión de TensorFlow:", tf.__version__)

# --- 1. Carga y Exploración de Datos (MNIST) ---
# Usamos el dataset MNIST para clasificación de dígitos (0-9)
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset_full, test_dataset = dataset['train'], dataset['test']

# Nombres de las clases en español
class_names = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis',
    'Siete', 'Ocho', 'Nueve'
]

# Conteo de ejemplos
num_train_examples_full = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

print(f"Total de ejemplos de entrenamiento: {num_train_examples_full}")
print(f"Total de ejemplos de prueba: {num_test_examples}")

# --- 2. Preprocesamiento de Datos y Conjunto de Validación ---

# Definir el tamaño del conjunto de validación (10% del entrenamiento)
VAL_FRACTION = 0.1
num_val_examples = int(num_train_examples_full * VAL_FRACTION)
num_train_examples = num_train_examples_full - num_val_examples

# Separar el conjunto de entrenamiento en entrenamiento y validación
train_dataset = train_dataset_full.skip(num_val_examples)
validation_dataset = train_dataset_full.take(num_val_examples)

print(f"Ejemplos finales para entrenamiento: {num_train_examples}")
print(f"Ejemplos para validación: {num_val_examples}")

# Normalizar: Función para escalar los valores de pixel de [0, 255] a [0, 1]
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255.0
    return images, labels

# Aplicar la normalización a los tres conjuntos
train_dataset = train_dataset.map(normalize)
validation_dataset = validation_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# --- 3. Configuración de Lotes (Batching) ---
BATCH_SIZE = 64 # Aumento el tamaño del lote de 32 a 64
SHUFFLE_BUFFER_SIZE = 10000 # Un buffer de mezcla más grande

# Aplicar shuffle, repeat y batch a los conjuntos
# El conjunto de entrenamiento se repite y se mezcla
train_dataset = train_dataset.repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# Los conjuntos de validación y prueba solo se agrupan en lotes
validation_dataset = validation_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# --- 4. Estructura del Modelo (Arquitectura) ---
model = tf.keras.Sequential([
    # Capa de aplanamiento para convertir la imagen 28x28x1 en un vector de 784
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    
    # Capas Dense (completamente conectadas) con activación ReLU
    tf.keras.layers.Dense(128, activation='relu', name='hidden_layer_1'), # Duplico el número de neuronas
    tf.keras.layers.Dropout(0.2), # Agrego Dropout para regularización
    tf.keras.layers.Dense(64, activation='relu', name='hidden_layer_2'),
    
    # Capa de salida con 10 unidades (una por clase) y activación Softmax para probabilidades
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
])

# Resumen del modelo
model.summary()

# --- 5. Configuración del Entrenamiento (Compilación) ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Uso un optimizador Adam con tasa de aprendizaje explícita
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Callbacks y Entrenamiento ---

# 6.1 Callbacks de Keras
# 1. Early Stopping: Detiene el entrenamiento si la métrica no mejora
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Monitorea la pérdida de validación
    patience=3,         # Espera 3 épocas sin mejora
    restore_best_weights=True # Restaura los mejores pesos encontrados
)

# 2. Model Checkpoint: Guarda el mejor modelo
checkpoint_filepath = './best_mnist_model.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False, # Guarda el modelo completo
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# 3. TensorBoard: Para visualización avanzada (logs)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Parámetros de entrenamiento
EPOCHS = 10 # Aumento el número de épocas, el EarlyStopping lo detendrá si es necesario
steps_per_epoch = math.ceil(num_train_examples / BATCH_SIZE)
validation_steps = math.ceil(num_val_examples / BATCH_SIZE)

# 6.2 Realizar el aprendizaje
print("\n--- Iniciando Entrenamiento del Modelo ---")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback] # Se añaden los callbacks
)

# --- 7. Evaluación y Guardado ---

# 7.1 Cargar el mejor modelo guardado (por el ModelCheckpoint)
print(f"\n--- Cargando el mejor modelo desde: {checkpoint_filepath} ---")
best_model = tf.keras.models.load_model(checkpoint_filepath)

# 7.2 Evaluar nuestro mejor modelo contra el dataset de pruebas
test_loss, test_accuracy = best_model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples / BATCH_SIZE)
)

print(f"\nResultado final en las pruebas: Precisión = {test_accuracy:.4f}, Pérdida = {test_loss:.4f}")

# --- 8. Visualización de Resultados ---

## 8.1 Historial de entrenamiento (Pérdida y Precisión)
print("\n--- Visualizando Historial de Entrenamiento ---")
plt.figure(figsize=(12, 4))

# Gráfico de Pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida (Entrenamiento)')
plt.plot(history.history['val_loss'], label='Pérdida (Validación)')
plt.title('Pérdida a lo largo de las Épocas')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Gráfico de Precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión (Entrenamiento)')
plt.plot(history.history['val_accuracy'], label='Precisión (Validación)')
plt.title('Precisión a lo largo de las Épocas')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()


## 8.2 Predicciones y Matriz de Confusión

# Obtener un lote de imágenes de prueba para predicción y visualización
for test_images, test_labels in test_dataset.take(1):
    test_images_numpy = test_images.numpy()
    test_labels_numpy = test_labels.numpy()
    predictions = best_model.predict(test_images_numpy)

# Función para graficar la imagen y la predicción
def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # Muestra la imagen. Se usa img[...,0] porque las imágenes son 28x28x1
    plt.imshow(img[...,0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue' # Correcto
    else:
        color = 'red'  # Incorrecto

    plt.xlabel(f"Predicción: {class_names[predicted_label]} ({int(100*np.max(predictions_array))}%)", color=color)

# Función para graficar el array de valores (probabilidades)
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#888888")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red') # Predicción (rojo)
    thisplot[true_label].set_color('blue')    # Etiqueta Verdadera (azul)

# Graficar un conjunto de predicciones (5 filas x 2 columnas)
num_rows = 5
num_cols = 2
num_images = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    # Imagen
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels_numpy, test_images_numpy)
    # Barras de probabilidad
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels_numpy)

plt.tight_layout()
plt.show()

## 8.3 Matriz de Confusión (para una evaluación más profunda)
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Obtener todas las etiquetas de prueba y todas las predicciones
all_test_labels = []
all_predictions = []

# Iterar sobre todo el dataset de prueba (sin mezclar ni repetir)
# Es necesario crear un nuevo dataset sin batching para poder obtener todas las etiquetas
test_dataset_full_unbatched = test_dataset.unbatch()
test_labels_list = list(test_dataset_full_unbatched.map(lambda image, label: label).as_numpy_iterator())
true_labels_full = np.array(test_labels_list)

# Obtener predicciones en el dataset con batching
predictions_full = best_model.predict(test_dataset, steps=math.ceil(num_test_examples / BATCH_SIZE))
predicted_labels_full = np.argmax(predictions_full, axis=1)

# Asegurarse de que el número de etiquetas y predicciones coincida
if len(true_labels_full) == len(predicted_labels_full):
    # Generar la Matriz de Confusión
    cm = confusion_matrix(true_labels_full, predicted_labels_full)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.title('Matriz de Confusión del Modelo MNIST')
    plt.show()
else:
    print("\nAdvertencia: La Matriz de Confusión no se pudo generar correctamente debido a una discrepancia en el número de etiquetas.")


"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()

logger.setLevel(logging.ERROR)


dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis',
    'Siete', 'Ocho', 'Nueve'
]

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

#Normalizar: Numeros de 0 a 255, que sean de 0 a 1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#Estructura de la red
model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28,1)),
	tf.keras.layers.Dense(64, activation=tf.nn.relu),
	tf.keras.layers.Dense(64, activation=tf.nn.relu),
	tf.keras.layers.Dense(10, activation=tf.nn.softmax) #para clasificacion
])

#Indicar las funciones a utilizar
model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

#Aprendizaje por lotes de 32 cada lote
BATCHSIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

#Realizar el aprendizaje
model.fit(
	train_dataset, epochs=5,
	steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE) #No sera necesario pronto
)

#Evaluar nuestro modelo ya entrenado, contra el dataset de pruebas
test_loss, test_accuracy = model.evaluate(
	test_dataset, steps=math.ceil(num_test_examples/32)
)

print("Resultado en las pruebas: ", test_accuracy)


for test_images, test_labels in test_dataset.take(1):
	test_images = test_images.numpy()
	test_labels = test_labels.numpy()
	predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_labels, images):
	predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img[...,0], cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("Prediccion: {}".format(class_names[predicted_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#888888")
	plt.ylim([0,1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

numrows=5
numcols=3
numimages = numrows*numcols

plt.figure(figsize=(2*2*numcols, 2*numrows))
for i in range(numimages):
	plt.subplot(numrows, 2*numcols, 2*i+1)
	plot_image(i, predictions, test_labels, test_images)
	plt.subplot(numrows, 2*numcols, 2*i+2)
	plot_value_array(i, predictions, test_labels)

plt.show()
