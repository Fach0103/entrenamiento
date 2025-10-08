import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

modelo = load_model("modelo_entrenado.h5")
clases = ["Plástico", "Papel", "Vidrio", "Orgánico", "Metal"]

def preparar_imagen(imagen):
    imagen = imagen.resize((224, 224))
    imagen_array = np.array(imagen) / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)
    return imagen_array

def predecir_residuo(imagen_array, modelo, clases):
    prediccion = modelo.predict(imagen_array)
    indice = np.argmax(prediccion)
    confianza = prediccion[0][indice]
    return clases[indice], confianza

st.set_page_config(page_title="Clasificador de Residuos", layout="centered")
st.title(" Clasificador de Residuos con IA")
st.markdown("Sube una imagen de un residuo para que el sistema lo clasifique automáticamente.")

archivo = st.file_uploader("📷 Sube una imagen", type=["jpg", "png", "jpeg"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    imagen_array = preparar_imagen(imagen)
    clase, confianza = predecir_residuo(imagen_array, modelo, clases)

    st.success(f"Residuo detectado: **{clase}**")
    st.info(f"Confianza del modelo: **{confianza*100:.2f}%**")
