import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================
st.set_page_config(
    page_title="🌸 Clasificador de Flores",
    page_icon="🌸",
    layout="centered"
)

# Clases en el mismo orden que el modelo fue entrenado
CLASSES    = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
IMG_SIZE   = 128
MODEL_PATH = "mejor_modelo_flores.keras"   # ← Pon aquí la ruta a tu modelo

# Colores por clase para visualización
CLASS_COLORS = {
    'daisy'      : '#FFD700',
    'dandelion'  : '#FFA500',
    'roses'      : '#FF6B9D',
    'sunflowers' : '#FF8C00',
    'tulips'     : '#9B59B6'
}

CLASS_EMOJI = {
    'daisy'      : '🌼',
    'dandelion'  : '🌻',
    'roses'      : '🌹',
    'sunflowers' : '🌻',
    'tulips'     : '🌷'
}

CLASS_NAMES_ES = {
    'daisy'      : 'Margarita',
    'dandelion'  : 'Diente de León',
    'roses'      : 'Rosa',
    'sunflowers' : 'Girasol',
    'tulips'     : 'Tulipán'
}

# ============================================================
# CARGA DEL MODELO (cacheado para no recargar en cada interacción)
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ============================================================
# PREPROCESAMIENTO DE IMAGEN
# ============================================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa la imagen para que coincida exactamente con
    el formato usado durante el entrenamiento:
    - Redimensiona a IMG_SIZE x IMG_SIZE
    - Convierte a array numpy
    - Normaliza a [0, 1]
    - Añade dimensión de batch
    """
    img = image.convert('RGB')                        # Asegurar 3 canales RGB
    img = img.resize((IMG_SIZE, IMG_SIZE))            # Redimensionar
    arr = np.array(img, dtype=np.float32) / 255.0    # Normalizar
    arr = np.expand_dims(arr, axis=0)                 # Shape: (1, 128, 128, 3)
    return arr

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
st.title("🌸 Clasificador de Flores")
st.markdown("**Reserva Botánica — Sistema de Identificación Automática**")
st.markdown("---")

# --- Sidebar con información ---
with st.sidebar:
    st.header("ℹ️ Sobre el Modelo")
    st.markdown("""
    **Arquitectura:** CNN desde cero
    
    **Clases que identifica:**
    """)
    for cls in CLASSES:
        st.markdown(f"- {CLASS_EMOJI[cls]} **{CLASS_NAMES_ES[cls]}** ({cls})")
    
    st.markdown("---")
    st.markdown("""
    **Preprocesamiento:**
    - Tamaño: 128×128 px
    - Normalización: [0, 1]
    
    **Arquitectura:**
    - 4 bloques Conv2D
    - BatchNormalization
    - GlobalAveragePooling
    - Dropout regularización
    """)

# --- Cargar modelo ---
model = load_model()

if model is None:
    st.error(f"""
    ❌ **Modelo no encontrado**
    
    Asegúrate de que el archivo `{MODEL_PATH}` esté en la misma carpeta que `app.py`.
    
    Si acabas de entrenar en Google Colab, descarga el archivo `mejor_modelo_flores.keras` 
    y colócalo junto a este archivo.
    """)
    st.stop()
else:
    st.success("✅ Modelo cargado correctamente")

# --- Carga de imagen ---
st.markdown("### 📤 Sube una imagen de flor")
uploaded_file = st.file_uploader(
    "Formatos aceptados: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    help="Sube una imagen clara de una flor para clasificarla"
)

# --- Predicción ---
if uploaded_file is not None:
    
    # Mostrar imagen original
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🖼️ Imagen cargada")
        st.image(image, use_column_width=True,
                 caption=f"Resolución original: {image.size[0]}×{image.size[1]} px")
    
    # Preprocesar y predecir
    with st.spinner("🔍 Analizando imagen..."):
        img_array   = preprocess_image(image)
        predictions = model.predict(img_array, verbose=0)[0]
    
    # Clase predicha
    pred_idx   = np.argmax(predictions)
    pred_class = CLASSES[pred_idx]
    pred_conf  = predictions[pred_idx] * 100
    pred_color = CLASS_COLORS[pred_class]
    pred_emoji = CLASS_EMOJI[pred_class]
    pred_name  = CLASS_NAMES_ES[pred_class]
    
    with col2:
        st.markdown("#### 🎯 Resultado")
        
        # Resultado principal destacado
        st.markdown(f"""
        <div style='
            background-color: {pred_color}22;
            border: 3px solid {pred_color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        '>
            <div style='font-size: 52px'>{pred_emoji}</div>
            <div style='font-size: 26px; font-weight: bold; color: #333'>
                {pred_name}
            </div>
            <div style='font-size: 14px; color: #666; margin-top: 4px'>
                ({pred_class})
            </div>
            <div style='font-size: 22px; font-weight: bold; 
                        color: {pred_color}; margin-top: 8px'>
                {pred_conf:.1f}% de confianza
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Nivel de confianza interpretado
        if pred_conf >= 80:
            st.success("🟢 Alta confianza — predicción confiable")
        elif pred_conf >= 60:
            st.warning("🟡 Confianza moderada — posible ambigüedad")
        else:
            st.error("🔴 Baja confianza — imagen poco clara o clase no representada")
    
    # --- Probabilidades por clase ---
    st.markdown("---")
    st.markdown("### 📊 Probabilidades por clase")
    st.markdown("La barra más larga indica la clase predicha por el modelo.")
    
    # Ordenar por probabilidad descendente
    sorted_idx   = np.argsort(predictions)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    bar_colors = [
        CLASS_COLORS[CLASSES[i]] if i == pred_idx else '#D0D0D0'
        for i in sorted_idx
    ]
    bar_labels = [
        f"{CLASS_EMOJI[CLASSES[i]]} {CLASS_NAMES_ES[CLASSES[i]]}"
        for i in sorted_idx
    ]
    bar_values = [predictions[i] * 100 for i in sorted_idx]
    
    bars = ax.barh(bar_labels, bar_values, color=bar_colors,
                   edgecolor='white', linewidth=1.5, height=0.6)
    
    # Etiquetas de porcentaje
    for bar, val in zip(bars, bar_values):
        ax.text(min(val + 1, 97), bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=12, fontweight='bold')
    
    # Resaltar la barra ganadora
    bars[0].set_edgecolor('#333333')
    bars[0].set_linewidth(2.5)
    
    ax.set_xlim(0, 105)
    ax.set_xlabel('Probabilidad (%)', fontsize=11)
    ax.set_title('Distribución de probabilidades por clase', fontsize=12, fontweight='bold')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # --- Tabla de probabilidades ---
    st.markdown("#### 📋 Detalle numérico")
    cols = st.columns(len(CLASSES))
    for i, col in enumerate(cols):
        idx   = sorted_idx[i]
        cls   = CLASSES[idx]
        prob  = predictions[idx] * 100
        emoji = CLASS_EMOJI[cls]
        name  = CLASS_NAMES_ES[cls]
        if idx == pred_idx:
            col.metric(
                label=f"{emoji} {name}",
                value=f"{prob:.1f}%",
                delta="← PREDICHO"
            )
        else:
            col.metric(
                label=f"{emoji} {name}",
                value=f"{prob:.1f}%"
            )
    
    # --- Análisis técnico expandible ---
    with st.expander("🔬 Ver detalles técnicos de la predicción"):
        st.markdown(f"""
        **Imagen procesada:**
        - Resolución original: {image.size[0]}×{image.size[1]} px
        - Resolución de entrada al modelo: {IMG_SIZE}×{IMG_SIZE} px
        - Canales: RGB (3 canales)
        - Normalización: valores divididos entre 255 → rango [0, 1]
        
        **Predicción:**
        - Clase predicha: `{pred_class}` (índice {pred_idx})
        - Probabilidad máxima: `{pred_conf:.4f}%`
        - Suma de probabilidades: `{predictions.sum():.6f}` (debe ser ≈ 1.0)
        
        **Vector de probabilidades (softmax output):**
        ```
        {dict(zip(CLASSES, [f'{p:.6f}' for p in predictions]))}
        ```
        """)

else:
    # Estado inicial — instrucciones
    st.markdown("---")
    st.info("""
    👆 **Sube una imagen** usando el botón de arriba para comenzar la clasificación.
    
    El modelo puede identificar las siguientes flores:
    """)
    
    cols = st.columns(5)
    for col, cls in zip(cols, CLASSES):
        col.markdown(f"""
        <div style='text-align:center; padding:15px; 
                    background:{CLASS_COLORS[cls]}22;
                    border-radius:10px; border:2px solid {CLASS_COLORS[cls]}'>
            <div style='font-size:32px'>{CLASS_EMOJI[cls]}</div>
            <div style='font-weight:bold; font-size:13px'>{CLASS_NAMES_ES[cls]}</div>
            <div style='font-size:11px; color:#666'>{cls}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:12px'>
    🌸 CNN desde cero — Proyecto Deep Learning | Reserva Botánica<br>
    Modelo entrenado con flower_photos dataset (TensorFlow)
</div>
""", unsafe_allow_html=True)
