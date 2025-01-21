from tools import cosine_similarity, clean_text
from embeddings import Embeddings
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import PyPDF2

# Configuración inicial de la página
st.set_page_config(page_title="Embedding Visualizer", layout="wide")

# Instancia de Embeddings
def draw_plot(embeddings, chunks, indexes):
    """Dibuja una gráfica 3D de embeddings usando Plotly."""
    
    if not embeddings:
        st.warning("No hay embeddings para visualizar.")
        return

    X = np.array([embedding[0] for embedding in embeddings])
    Y = np.array([embedding[1] for embedding in embeddings])
    Z = np.array([embedding[2] for embedding in embeddings])

    # Asignación de colores
    color = ['black' if i not in indexes else 'green' for i in range(len(embeddings))]
    if len(color) > 0:
        color[-1] = 'red'

    # Crear figura 3D con Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        mode='markers',
        marker=dict(size=6, color=color, opacity=0.6),
        text=[chunks[i] for i in range(len(chunks))],
        hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<br>'
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title="Visualización 3D de Embeddings",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig)

def procesar_datos(pdf_file, texto):
    """Procesa los datos del PDF y del texto ingresado."""

    pdf_text = ""

    # Procesar archivo PDF
    if pdf_file is not None:
        with st.spinner("Procesando PDF..."):
            st.success(f"📄 Archivo PDF recibido: **{pdf_file.name}**")
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                pdf_text = '\n'.join([page.extract_text() or "" for page in pdf_reader.pages])
                pdf_text = clean_text(pdf_text)
                
                with st.expander("📃 Texto extraído del PDF"):
                    st.text_area("Texto del PDF", pdf_text, height=200)
                    
            except Exception as e:
                st.error("❌ Error al procesar el PDF")
                st.exception(e)
    else:
        st.warning("⚠ No se subió ningún archivo PDF.")

    # Procesar texto ingresado
    if texto:
        st.success(f"📝 Texto ingresado: **{texto}...**")
    else:
        st.warning("⚠ No se ha ingresado texto.")

    # Generar embeddings y graficar
    try:
        embeddings, chunks = embedding.get_embedding_and_chunks(pdf_text, texto, chunk_size, chunk_overlap)
        embeddings = embeddings.tolist()
        if texto:
            chunks.append(texto)

        if not embeddings:
            st.warning("⚠ No se generaron embeddings.")
            return

        indexes = cosine_similarity(embeddings, result_size)

        with st.expander("📌 Resultados de similitud"):
            for index in indexes:
                st.markdown(f"- **{chunks[index]}**")

        draw_plot(embeddings, chunks, indexes)
    
    except Exception as e:
        st.error("❌ Error al calcular embeddings.")
        st.exception(e)

# ------------------------- UI -------------------------
st.title("📊 Embedding Visualizer")

# Sidebar para subir archivos y escribir texto
st.sidebar.header("📂 Entrada de Datos")
pdf_file = st.sidebar.file_uploader("📎 Sube un archivo PDF", type=["pdf"])
texto = st.sidebar.text_area("✍ Introduce un texto", placeholder="Escribe aquí...")
chunk_size = st.sidebar.number_input("Tamaño de los chunks", min_value=1, value=200)
chunk_overlap = st.sidebar.number_input("Superposición de los chunks", min_value=0, value=50)
result_size = st.sidebar.number_input("Número de resultados", min_value=1, value=5)

embedding = Embeddings()

if st.sidebar.button("🚀 Analizar"):
    procesar_datos(pdf_file, texto)
