# Detección de Neumonía en Rayos X con CNN

Proyecto de clasificación de imágenes médicas usando redes neuronales convolucionales para identificar neumonía en radiografías de tórax.

## Descripción

El objetivo de este proyecto es desarrollar un modelo de clasificación de imágenes médicas que permita identificar la presencia de neumonía en radiografías de tórax. Para ello, se ha utilizado un dataset de imágenes de rayos X de tórax de pacientes con neumonía y sin neumonía. Usaremos redes neuronales convolucionales (CNN) con tecnologías como TensorFlow y Keras para entrenar el modelo.

**Dataset utilizado**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data).

## Instalación de dependencias y recomendaciones previas

> ⚠️ Es necesario tener Python 3.12 o inferior para ejecutar el proyecto
>
> ⚠️ Recomendamos encarecidamente ejecutar el proyecto en Linux

Para trabajar en un entorno aislado, lo mejor (aunque opcional) es crear un entorno virtual ejecutando los siguientes comandos:

```
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate  # Windows (depende la version de python, podria ser: .\venv\bin\activate
```

Seguidamente debemos instalar las dependencias necesarias presentes en el archivo `requirements.txt:`

```
pip install -r requirements.txt
```

## Ejecución del cuaderno Jupyter

El archivo `neumoniaNotebook.ipynb` es un cuaderno Jupyter organizado en secciones donde se explica que hace cada parte del codigo. Para ejecutarlo es tan simple como abrir el archivo con alguna herramienta como **VSCode** para visualizar su contenido y ejecutarlo. Alternativamente, puedes abrir el cuaderno en un navegador gracias a **Jupyter Lab** siguiendo estos pasos:

1. Lanza un servidor Jupyter ejecutando este comando:

   ```
   jupyter lab
   ```

   En principio debería abrirse automaticamente en el navegador, pero si no es así, continuar con los pasos.
2. Para acceder al servidor, ejecuta este comando en otra terminal:

   ```
   jupyter server list
   ```
3. Deberías ver una url localhost, ese es tu servidor. Haz `ctrl + click` y se te abrirá el cuaderno en el navegador, donde podrás ejecutarlo por secciones o todo de golpe.

Si no quieres ejecutar el cuaderno, puedes ejecutar `train_and_test_model.py`, que es esencialmente el cuaderno pero sin tanto detalle en los outputs. Para ello ejecuta el comando:

```
python3 train_and_test_model.py
```

Una vez ejecutado el cuaderno o el archivo python, se habra creado un modelo `modelo_neumonia.h5` si no se había creado previamente. Este es el modelo ya entrenado y listo para pasarle imágenes.

## Iniciar interfaz gráfica

Para iniciar la interfaz basta con ejecutar el comando:

```
streamlit run app.py
```
