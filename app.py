import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Contador de Tetrastichus")

st.title("🦟 Contador de Tetrastichus")

area_min = st.slider("Área mínima", 1, 100, 10)
area_max = st.slider("Área máxima", 10, 500, 100)

arquivo = st.file_uploader(
    "Envie uma foto da placa",
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

    st.write("Resolução:", img.shape)

    h, w = img.shape[:2]

    metade_h = h // 2
    metade_w = w // 2

    quadrantes = [
        ("Q1", img[0:metade_h, 0:metade_w]),
        ("Q2", img[0:metade_h, metade_w:w]),
        ("Q3", img[metade_h:h, 0:metade_w]),
        ("Q4", img[metade_h:h, metade_w:w]),
    ]

    total_geral = 0

    resultado_final = img.copy()

    st.subheader("Contagem por quadrante")

    for nome, quad in quadrantes:

        gray = cv2.cvtColor(
            quad,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        blur = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        _, mask = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((2, 2), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contagem_quad = 0

        for c in contours:

            area = cv2.contourArea(c)

            if area_min <= area <= area_max:

                x, y, ww, hh = cv2.boundingRect(c)

                contagem_quad += 1

                if nome == "Q1":
                    offset_x = 0
                    offset_y = 0

                elif nome == "Q2":
                    offset_x = metade_w
                    offset_y = 0

                elif nome == "Q3":
                    offset_x = 0
                    offset_y = metade_h

                else:
                    offset_x = metade_w
                    offset_y = metade_h

                cv2.rectangle(
                    resultado_final,
                    (x + offset_x, y + offset_y),
                    (x + ww + offset_x, y + hh + offset_y),
                    (0, 0, 255),
                    2
                )

        total_geral += contagem_quad

        st.write(f"{nome}: {contagem_quad} insetos")

    st.success(f"Total estimado: {total_geral}")

    st.subheader("Foto Original")

    st.image(
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        ),
        use_container_width=True
    )

    st.subheader("Detecção")

    st.image(
        cv2.cvtColor(
            resultado_final,
            cv2.COLOR_BGR2RGB
        ),
        use_container_width=True
    )
