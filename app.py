import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import json
import os

st.set_page_config(page_title="Detector de Neumonía", layout="wide")
st.title("🩺 Detector de Neumonía en Radiografías")

# Selector de modelo en la barra lateral
st.sidebar.header("Configuración")
model_choice = st.sidebar.selectbox(
    "Selecciona el modelo de predicción",
    [
        "CNN (entrenada desde cero)",
        "MobileNetV2 + Random Forest",
        "MobileNetV2 Fine-tuned"
    ]
)

MODEL_KEYS = {
    "CNN (entrenada desde cero)":    "CNN",
    "MobileNetV2 + Random Forest":   "MobileNetV2 + RF",
    "MobileNetV2 Fine-tuned":        "MobileNetV2 Fine-tuned"
}

# Mostrar métricas del modelo seleccionado si el JSON existe
if os.path.exists('model_metrics.json'):
    with open('model_metrics.json') as f:
        all_metrics = json.load(f)
    metric_key = MODEL_KEYS[model_choice]
    if metric_key in all_metrics:
        m = all_metrics[metric_key]
        st.sidebar.markdown("### Métricas en test set")
        st.sidebar.metric("Accuracy", f"{m['accuracy']:.2%}")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Precisión", f"{m['precision']:.2%}")
        c2.metric("Recall",    f"{m['recall']:.2%}")
        st.sidebar.metric("F1-Score", f"{m['f1']:.2%}")
        st.sidebar.caption(
            "**Recall** mide la proporción de casos de neumonía correctamente detectados. "
            "En diagnóstico médico es prioritario sobre la precisión."
        )


@st.cache_resource
def load_cnn():
    return load_model('modelo_neumonia.h5')

@st.cache_resource
def load_mobilenet_extractor():
    return load_model('modelo_mobilenet_features.h5')

@st.cache_resource
def load_rf():
    return joblib.load('modelo_rf.pkl')

@st.cache_resource
def load_finetuned():
    return load_model('modelo_mobilenet_finetuned.h5')


uploaded_file = st.file_uploader("Sube una radiografía de tórax...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción según modelo elegido
        if model_choice == "CNN (entrenada desde cero)":
            prob_neumonia = float(load_cnn().predict(img_array, verbose=0)[0][0])

        elif model_choice == "MobileNetV2 + Random Forest":
            features = load_mobilenet_extractor().predict(img_array, verbose=0)
            rf = load_rf()
            if hasattr(rf, 'predict_proba'):
                prob_neumonia = float(rf.predict_proba(features)[0][1])
            else:
                prob_neumonia = float(rf.predict(features)[0])

        else:  # MobileNetV2 Fine-tuned
            prob_neumonia = float(load_finetuned().predict(img_array, verbose=0)[0][0])

        prob_normal = 1 - prob_neumonia

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Imagen analizada", width=300)
            st.markdown(f"**Modelo usado:** {model_choice}")
            if prob_neumonia > 0.5:
                st.error(f"🚨 **Neumonía detectada** — probabilidad: {prob_neumonia:.2%}")
            else:
                st.success(f"✅ **Sin neumonía** — probabilidad normal: {prob_normal:.2%}")

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.barh(['Normal', 'Neumonía'], [prob_normal, prob_neumonia], color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilidad")
            ax.bar_label(bars, fmt='%.2f', padding=4)
            ax.set_title(f"Resultado de predicción")
            fig.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")

st.markdown("---")
st.caption("**Nota:** Esta herramienta es solo orientativa. Consulte siempre a un profesional médico para validar cualquier diagnóstico.")
