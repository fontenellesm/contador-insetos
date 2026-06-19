import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# =========================
# CONFIGURAÇÃO VISUAL
# =========================
st.set_page_config(page_title="Insetos", layout="wide")

st.markdown("""
<style>

[data-testid="stMetric"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stMetricValue"] {
    color: #00ff88 !important;
    font-size: 28px !important;
}

[data-testid="stMetricLabel"] {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TÍTULO
# =========================
st.title("🐞 Contador de Insetos - Viabilidade")

# =========================
# CONTROLES
# =========================
col1, col2 = st.columns(2)

with col1:
    area_min = st.slider("Área mínima", 1, 50, 3)

with col2:
    area_max = st.slider("Área máxima", 10, 500, 200)

foto = st.camera_input("📷 Tire uma foto da placa")

# =========================
# FUNÇÃO EXCEL
# =========================
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='insetos')
    return output.getvalue()

# =========================
# PROCESSAMENTO
# =========================
if foto is not None:

    file_bytes = np.frombuffer(foto.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 30, 90)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total = 0
    img_result = img.copy()

    areas = []

    for c in contours:

        area = cv2.contourArea(c)

        if area_min < area < area_max:

            x, y, w, h = cv2.boundingRect(c)

            aspect = w / float(h) if h > 0 else 0

            if 0.2 < aspect < 5:

                total += 1
                areas.append(area)

                cv2.rectangle(
                    img_result,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

    # =========================
    # DASHBOARD
    # =========================
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🐞 Insetos detectados", total)

    with c2:
        st.metric("📏 Área mínima", area_min)

    with c3:
        st.metric("📏 Área máxima", area_max)

    st.markdown("---")

    # =========================
    # GRÁFICO
    # =========================
    if len(areas) > 0:

        st.subheader("📊 Distribuição das áreas")

        fig, ax = plt.subplots()
        ax.hist(areas, bins=10, color="green")
        ax.set_title("Distribuição dos insetos detectados")
        ax.set_xlabel("Área (pixels)")
        ax.set_ylabel("Quantidade")

        st.pyplot(fig)

    # =========================
    # EXCEL EXPORT
    # =========================
    df = pd.DataFrame({
        "Inseto": list(range(1, total + 1)),
        "Area": areas if len(areas) == total else [0]*total
    })

    excel = to_excel(df)

    st.download_button(
        "📁 Baixar Excel",
        data=excel,
        file_name="insetos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")

    # =========================
    # IMAGENS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    with col2:
        st.subheader("🔎 Edges")
        st.image(edges, clamp=True)

    with col3:
        st.subheader("🎯 Resultado")
        st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
