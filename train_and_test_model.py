# Detección de Neumonía en Radiografías con Deep Learning y Random Forest
# Este script explora dos enfoques para la detección automática de neumonía:
# - Una red convolucional (CNN) entrenada desde cero.
# - Un clasificador Random Forest sobre características extraídas por MobileNetV2.

# 1. Importación de librerías necesarias
# Importamos todas las librerías necesarias para el tratamiento de imágenes, construcción y entrenamiento de modelos con Keras, extracción de características con MobileNetV2, clasificación tradicional con Scikit-learn y visualización con matplotlib y seaborn.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# 2. Configuración de parámetros y rutas
# Definimos las rutas a los conjuntos de datos (entrenamiento, validación y prueba), así como parámetros globales del proyecto como el tamaño de las imágenes, el batch size y el número de épocas de entrenamiento.

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')
img_width, img_height = 150, 150
batch_size = 32
epochs = 10

# 3. Generación de datos
# Creamos generadores para cargar imágenes desde disco y procesarlas de forma automática.
# Aplicamos técnicas de aumento de datos para robustecer el modelo durante el entrenamiento, y normalizamos los píxeles dividiendo entre 255.
# Desactivamos el barajado (`shuffle=False`) en los conjuntos de entrenamiento y prueba para mantener el orden de las muestras, importante para la correspondencia con sus etiquetas.

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary', shuffle=False)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')
test_generator = val_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
    batch_size=1, class_mode='binary', shuffle=False)

# 4. Entrenamiento o carga del modelo CNN
# Si existe el archivo `modelo_neumonia.h5`, cargamos el modelo ya entrenado.
# En caso contrario, construimos una red neuronal convolucional desde cero, la entrenamos con los datos proporcionados y la guardamos para futuros usos.

if os.path.exists('modelo_neumonia.h5'):
    print("Cargando modelo existente...")
    model = load_model('modelo_neumonia.h5')
else:
    print("Entrenando nuevo modelo CNN...")
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator, validation_steps=val_generator.samples // batch_size,
        epochs=epochs)
    model.save('modelo_neumonia.h5')

# 5. Predicción con la CNN
# Realizamos predicciones sobre las imágenes del conjunto de prueba usando el modelo CNN.
# Convertimos las probabilidades en etiquetas binarias (0 o 1) comparándolas con un umbral de 0.5.

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)

# 6. Extracción de características con MobileNetV2
# MobileNetV2 se utiliza como extractor de características sin reentrenamiento (transfer learning).
# A partir de cada imagen, genera un vector numérico que resume patrones visuales relevantes aprendidos en ImageNet.

mobilenet = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False,
                          weights='imagenet', pooling='avg')
train_features = mobilenet.predict(train_generator)
test_features = mobilenet.predict(test_generator)

# 7. Clasificación con Random Forest
# Entrenamos un modelo `RandomForestClassifier` utilizando los vectores de características generados por MobileNetV2.
# Activamos `class_weight='balanced'` para compensar el desbalance entre clases.
# Después usamos este modelo para predecir sobre las imágenes de prueba.

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rf.fit(train_features, train_generator.classes)
y_pred_rf = rf.predict(test_features)

# 8. Comparación de resultados entre modelos
# Calculamos métricas de rendimiento para ambos enfoques (CNN y MobileNet + RF), incluyendo:
# - Exactitud global (`accuracy`)
# - Distribución de clases predichas
# - Informe detallado de clasificación con precisión, recall y F1-score

acc_cnn = accuracy_score(test_generator.classes, predicted_classes)
acc_rf = accuracy_score(test_generator.classes, y_pred_rf)
print(f"\nExactitud modelo CNN manual: {acc_cnn:.2f}")
print(f"Exactitud modelo MobileNet + Random Forest: {acc_rf:.2f}\n")
print("Distribución real de clases en test:", np.bincount(test_generator.classes))
print("Distribución de predicciones con RF:", np.bincount(y_pred_rf))
print("=== Informe CNN manual ===")
print(classification_report(test_generator.classes, predicted_classes))
print("=== Informe MobileNet + RF ===")
print(classification_report(test_generator.classes, y_pred_rf))

# 9. Visualización de la matriz de confusión
# Mostramos gráficamente los resultados de clasificación con una matriz de confusión para cada modelo.
# Esto permite identificar en qué tipo de casos se producen más errores (falsos positivos o negativos).

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

plot_confusion(test_generator.classes, predicted_classes, "Matriz de confusión - CNN manual")
plot_confusion(test_generator.classes, y_pred_rf, "Matriz de confusión - MobileNet + RF")
