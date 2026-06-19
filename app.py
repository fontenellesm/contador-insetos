import streamlit as st
import cv2
import numpy as np

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Contador de Viabilidade",
    page_icon="🦟",
    layout="wide"
)

# =========================
# CSS PROFISSIONAL
# =========================
st.markdown("""
<style>

h1 {
    text-align: center;
    color: #00ff88;
    font-size: 40px;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TÍTULO
# =========================
st.title("🦟 Contador de Viabilidade")

# =========================
# SIDEBAR (CONTROLES)
# =========================
with st.sidebar:
    st.header("⚙️ Configurações")

    area_min = st.slider("Área mínima", 1, 50, 3)
    area_max = st.slider("Área máxima", 10, 500, 200)

    st.info("Ajuste os filtros para melhorar a precisão da contagem.")

# =========================
# CÂMERA
# =========================
foto = st.camera_input("📷 Tire uma foto da placa")

# =========================
# FUNÇÃO DE PROCESSAMENTO
# =========================
def detectar(img, area_min, area_max):

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
                    (0, 255, 0),
                    2
                )

    return total, img_result, edges

# =========================
# EXECUÇÃO
# =========================
if foto is not None:

    file_bytes = np.frombuffer(foto.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    total, resultado, edges = detectar(img, area_min, area_max)

    # =========================
    # MÉTRICAS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🦟 Insetos detectados", total)

    with col2:
        st.metric("📏 Área mínima", area_min)

    with col3:
        st.metric("📏 Área máxima", area_max)

    st.markdown("---")

    # =========================
    # IMAGENS
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("🔎 Detecção")
        st.image(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("🧪 Edges (o que foi detectado)")
    st.image(edges, clamp=True, use_container_width=True)
