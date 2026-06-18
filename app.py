import streamlit as st
import cv2
import numpy as np

st.title("🐞 Contador de Insetos")

area_min = st.slider("Área mínima", 1, 100, 5)
area_max = st.slider("Área máxima", 10, 1000, 300)

arquivo = st.file_uploader(
    "Envie uma foto da placa (30x30 cm)",
    type=["jpg", "jpeg", "png"]
)

if arquivo is not None:

    file_bytes = np.frombuffer(
        arquivo.read(),
        np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Converter para cinza
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Melhorar contraste
    gray = cv2.equalizeHist(gray)

    # Suavização
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Detectar regiões escuras
    _, mask = cv2.threshold(
        blur,
        80,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Limpeza de ruído
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Contornos
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total = 0

    resultado = img.copy()

    for c in contours:

        area = cv2.contourArea(c)

        if area_min <= area <= area_max:

            x, y, w, h = cv2.boundingRect(c)

            total += 1

            cv2.rectangle(
                resultado,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

    st.success(f"Total encontrado: {total}")

    st.subheader("Foto Original")
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    st.subheader("Máscara")
    st.image(
        mask,
        clamp=True,
        use_container_width=True
    )

    st.subheader("Insetos Detectados")
    st.image(
        cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )COLOR_BGR2RGB))
