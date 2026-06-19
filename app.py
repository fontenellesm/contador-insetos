import streamlit as st
import cv2
import numpy as np

st.title("🦟 Contador de Pontos Pretos (Insetos)")

area_min = st.slider("Área mínima", 1, 50, 3)
area_max = st.slider("Área máxima", 5, 200, 60)

foto = st.camera_input("Tire uma foto da placa")

if foto is not None:

    file_bytes = np.frombuffer(foto.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # melhora contraste
    clahe = cv2.createCLAHE(2.0, (8,8))
    gray = clahe.apply(gray)

    # remove textura leve
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # 🔥 pega SOMENTE pixels realmente escuros
    _, mask = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)

    # remove ruído pequeno
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # junta pontinhos do mesmo inseto
    mask = cv2.dilate(mask, kernel, iterations=1)

    # componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    resultado = img.copy()
    total = 0

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # 🔥 FILTRO MAIS ESTRITO (SÓ PONTOS PEQUENOS)
        if area_min <= area <= area_max:

            # evita textura (muito alongado não é inseto)
            if 0.5 < (w / (h + 1e-5)) < 2.5:

                total += 1

                cv2.rectangle(
                    resultado,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    1
                )

    st.success(f"Total de insetos: {total}")

    st.subheader("Máscara (só pontos pretos)")
    st.image(mask, clamp=True)

    st.subheader("Resultado")
    st.image(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
