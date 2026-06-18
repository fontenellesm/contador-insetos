import streamlit as st
import cv2
import numpy as np

st.title("🐞 Contador de Pontos Pretos (versão robusta)")

arquivo = st.file_uploader("Envie a imagem da placa", type=["jpg","jpeg","png"])

if arquivo is not None:

    file_bytes = np.frombuffer(arquivo.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 melhora contraste local (MUITO importante)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # leve blur
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # 🔥 threshold adaptativo (melhor que OTSU aqui)
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # limpeza
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    mask = cv2.dilate(mask, kernel, iterations=1)

    # =========================
    # CONTAGEM POR COMPONENTES
    # =========================
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    total = 0
    img_result = img.copy()

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # 🔥 ajuste crítico para 1mm
        if 2 <= area <= 60:

            total += 1

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            cv2.rectangle(
                img_result,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                1
            )

    st.success(f"Total de pontos pretos: {total}")

    st.subheader("Máscara gerada")
    st.image(mask, clamp=True)

    st.subheader("Resultado")
    st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
