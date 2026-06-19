import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Contador de Tetrastichus")

st.title("🦟 Contador de Insetos por Quadrantes")

area_min = st.slider(
"Área mínima",
1,
100,
5
)

area_max = st.slider(
"Área máxima",
10,
300,
80
)

st.markdown("### Tire 4 fotos da placa")

q1 = st.camera_input("Quadrante 1 (superior esquerdo)")
q2 = st.camera_input("Quadrante 2 (superior direito)")
q3 = st.camera_input("Quadrante 3 (inferior esquerdo)")
q4 = st.camera_input("Quadrante 4 (inferior direito)")

def contar_insetos(foto, area_min, area_max):

```
file_bytes = np.frombuffer(
    foto.getvalue(),
    np.uint8
)

img = cv2.imdecode(
    file_bytes,
    cv2.IMREAD_COLOR
)

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

gray = clahe.apply(gray)

gray = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)

mask = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

kernel = np.ones(
    (2, 2),
    np.uint8
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

resultado = img.copy()

total = 0

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area_min <= area <= area_max:

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        total += 1

        cv2.rectangle(
            resultado,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            1
        )

return total, resultado, mask
```

if q1 and q2 and q3 and q4:

```
total1, img1, mask1 = contar_insetos(
    q1,
    area_min,
    area_max
)

total2, img2, mask2 = contar_insetos(
    q2,
    area_min,
    area_max
)

total3, img3, mask3 = contar_insetos(
    q3,
    area_min,
    area_max
)

total4, img4, mask4 = contar_insetos(
    q4,
    area_min,
    area_max
)

total_geral = (
    total1 +
    total2 +
    total3 +
    total4
)

st.success(
    f"Total estimado: {total_geral}"
)

st.write(f"Q1: {total1}")
st.write(f"Q2: {total2}")
st.write(f"Q3: {total3}")
st.write(f"Q4: {total4}")

col1, col2 = st.columns(2)

with col1:
    st.image(
        cv2.cvtColor(
            img1,
            cv2.COLOR_BGR2RGB
        ),
        caption=f"Q1 ({total1})"
    )

    st.image(
        cv2.cvtColor(
            img3,
            cv2.COLOR_BGR2RGB
        ),
        caption=f"Q3 ({total3})"
    )

with col2:
    st.image(
        cv2.cvtColor(
            img2,
            cv2.COLOR_BGR2RGB
        ),
        caption=f"Q2 ({total2})"
    )

    st.image(
        cv2.cvtColor(
            img4,
            cv2.COLOR_BGR2RGB
        ),
        caption=f"Q4 ({total4})"
    )
```
