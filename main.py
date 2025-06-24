import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

#Carrega modelo pré treinado
model = YOLO("best.pt")

st.set_page_config(page_title="Ponderada Periquito", layout="centered")

# Título com degradê azul → rosa - Criado pelo GPT pra ficar bonitinho :)
st.markdown(
    """
    <h1 style='text-align: center;
               font-size: 3em;
               background: -webkit-linear-gradient(90deg, #0099ff, #ff00ff);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;'>
        Mega Classificador de Imagens Periquito 1.1
    </h1>
    """,
    unsafe_allow_html=True
)

if "image_loaded" not in st.session_state:
    st.session_state["image_loaded"] = False
#Se não tiver imagem carregada já, pode uploadar imagem
if not st.session_state["image_loaded"]:
    uploaded_file = st.file_uploader("Upar imagem", type=["png", "jpg", "jpeg", "webp"])
    camera_image = st.camera_input("Ou tire uma foto")

#Depois de muito erro, converter a imagem pra CV2 ajudou bastante a resolver eventuais problemas de "Formato não suportado"
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        st.session_state["image_data"] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        st.session_state["image_loaded"] = True
        st.rerun()
    elif camera_image is not None:
        image_data = camera_image.getvalue()        
        nparr = np.frombuffer(image_data, np.uint8)
        st.session_state["image_data"] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        st.session_state["image_loaded"] = True
        st.rerun()
    
else:
    image_data = st.session_state.get("image_data")

if st.session_state["image_loaded"]:
    # Decodificar e mostrar a imagem
    image_data = st.session_state["image_data"]
    img_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, use_container_width=True, caption="Imagem Capturada/Enviada")
    result = model.predict(source=image_data, save=False, conf=0.25, show=False)
    st.markdown("### Resultado da Detecção:")
    st.write(f"Classe: ",model.names[result[0].probs.top1])
    st.write(f"Confiança: {result[0].probs.top1conf * 100:.2f}%")
    st.session_state["image_data"] = image_data
else:
    st.info("Por favor, faça o upload de uma imagem ou tire uma foto para começar.")
    st.text("Desenvolvido pelo Periquito, com amor, carinho, pressa, um toque de procrastinação e muita cafeína")