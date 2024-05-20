import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Ruta del modelo preentrenado
MODEL_PATH = './models/test_inception_v3_frozen_model.h5'

# Función para realizar la predicción usando el modelo
def model_prediction(img_path, model):
    # Carga la imagen y redimensiona a (125, 125) según lo esperado por el modelo
    img = image.load_img(img_path, target_size=(125, 125))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normaliza la imagen

    preds = model.predict(img)
    return preds

def main():
    # Cargar el modelo
    model = load_model(MODEL_PATH)

    # Título de la aplicación
    html_temp = """
    <h1 style="color:#0D8ABC;text-align:center;">Rede Neuronal Lentes</h1>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Estilos CSS personalizados
    st.markdown(
        """
        <style>
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin-top: 20px;
        }
        .uploaded-image {
            border: 2px solid #0D8ABC;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            max-width: 80%;
            height: auto;
        }
        .prediction-container {
            border: 2px solid #0D8ABC;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .prediction-text {
            font-size: 1.5em;
            color: #0D8ABC;
            margin-top: 10px;
        }
        .caption {
            font-size: 1.2em;
            color: #666;
            margin-top: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Subir una imagen
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Guardar la imagen subida temporalmente
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Mostrar la imagen subida y la predicción
        st.image(uploaded_file, caption='Imagen subida', use_column_width=True)

        st.write("Clasificando...")
        # Realizar la predicción
        prediction = model_prediction("temp.jpg", model)
        st.markdown(
            f'<div class="prediction-container">'
            f'<p class="prediction-text">Predicción: {"Sin Lentes" if prediction[0][0] > 0.5 else "Con Lentes"}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()
