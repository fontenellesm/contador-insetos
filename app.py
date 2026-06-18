import streamlit as st
import cv2
import numpy as np

st.title("🦟 Contador de Insetos-Viabilidade ") 

area_min = st.slider("Área mínima", 1, 50, 3)
area_max = st.slider("Área máxima", 10, 500, 200)

foto = st.camera_input("Tire uma foto da placa")

if foto is not None:

    file_bytes = np.frombuffer(foto.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 melhora contraste (IMPORTANTE aqui)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 🔥 usa Canny (MELHOR PARA SEU CASO)
    edges = cv2.Canny(blur, 30, 90)

    # fecha pequenos buracos
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total = 0
    img_result = img.copy()

    for c in contours:

        area = cv2.contourArea(c)

        if area_min < area < area_max:

            x, y, w, h = cv2.boundingRect(c)

            aspect = w / float(h) if h > 0 else 0

            if 0.2 < aspect < 5:

                total += 1

                cv2.rectangle(
                    img_result,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    1
                )

    st.success(f"Total estimado: {total}")

    st.subheader("Original")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    st.subheader("Edges (detecção)")
    st.image(edges, clamp=True)

    st.subheader("Resultado")
    st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
