import streamlit as st
import cv2
import numpy as np

st.title("🐞 Contador de Pontos Pretos (ajustado para 1mm)")

arquivo = st.file_uploader("Envie a imagem da placa", type=["jpg","jpeg","png"])

if arquivo is not None:

    file_bytes = np.frombuffer(arquivo.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.write("Resolução:", img.shape)

    # =========================
    # PRÉ-PROCESSAMENTO
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # normaliza contraste (ESSENCIAL)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # =========================
    # DETECÇÃO DE PIXEIS ESCUROS
    # =========================
    # pega tudo que é mais escuro que o fundo
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # limpa ruído leve
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # junta pixels próximos (IMPORTANTÍSSIMO)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # =========================
    # CONTAGEM POR COMPONENTES
    # =========================
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    total = 0
    img_result = img.copy()

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # filtros para remover sujeira e borda
        if 3 < area < 80:

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

    st.success(f"Total estimado: {total}")

    st.subheader("Máscara")
    st.image(mask, clamp=True)

    st.subheader("Resultado")
    st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
