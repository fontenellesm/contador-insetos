import streamlit as st
import cv2
import numpy as np

# =========================
# CONFIGURAÇÃO
# =========================
st.set_page_config(
    page_title="Contador de Viabilidade",
    page_icon="🦟",
    layout="wide"
)

# =========================
# ESTILO
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
    padding: 12px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TÍTULO
# =========================
st.title("🦟 Contador de Viabilidade")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Ajustes")

    area_min = st.slider("Área mínima", 1, 50, 5)
    area_max = st.slider("Área máxima", 5, 300, 80)

    st.info("Mantenha distância e luz constantes.")

# =========================
# CÂMERA
# =========================
foto = st.camera_input("📷 Tire uma foto da placa")

# =========================
# FUNÇÃO DE DETECÇÃO
# =========================
def detectar(img, area_min, area_max):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # melhora contraste local
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # suaviza sem destruir pontos
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # pega pontos escuros reais
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3
    )

    # limpa ruído
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # conecta pequenos pontos do inseto
    mask = cv2.dilate(mask, kernel, iterations=1)

    # componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    resultado = img.copy()
    total = 0

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        aspect = w / (h + 1e-5)

        if area_min <= area <= area_max:

            # filtro de formato (remove sujeira alongada)
            if 0.4 < aspect < 2.5:

                total += 1

                cv2.rectangle(
                    resultado,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

    return total, resultado, mask

# =========================
# EXECUÇÃO
# =========================
if foto is not None:

    file_bytes = np.frombuffer(foto.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    total, resultado, mask = detectar(img, area_min, area_max)

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

    st.subheader("🧪 Máscara (o que foi detectado)")
    st.image(mask, clamp=True, use_container_width=True)
