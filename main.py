from sklearn.metrics.pairwise import cosine_similarity
from embeddings import Embeddings
import plotly.graph_objects as go
from constants import RESULT_SIZE
import streamlit as st
import numpy as np
import PyPDF2
import re

embedding = Embeddings()

def clean_text(pdf_text):
    pdf_text = pdf_text.replace('\n', ' ')
    pdf_text = re.sub(r'\s+', ' ', pdf_text)
    pdf_text = pdf_text.strip()

    pdf_text = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ\s.,;:¡¿]+', ' ', pdf_text)
    return pdf_text

def draw_plot(embeddings, chunks, indexes):
    # Extraer las coordenadas X, Y, Z de los embeddings
    X = np.array([embedding[0] for embedding in embeddings])
    Y = np.array([embedding[1] for embedding in embeddings])
    Z = np.array([embedding[2] for embedding in embeddings])

    color = ['black' if i not in indexes else 'green' for i in range(len(embeddings))]
    color[-1] = 'red'

    # Crear la gráfica 3D con Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=X.flatten(), 
        y=Y.flatten(), 
        z=Z.flatten(),
        mode='markers',
        marker=dict(
            size=6, 
            color=color, 
            opacity=0.6
        ),
        text=[chunks[i] for i in range(len(chunks))],
        hovertemplate='X: %{x}<br>' + 
            'Y: %{y}<br>' +
            'Z: %{z}<br>'
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title='Visualización 3D de Embeddings',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig)

def calculate_nearest_point(embeddings):
    points = embeddings[0:-2]
    text = embeddings[-1]

    res = [cosine_similarity([point], [text])[0][0] for point in points]

    points_with_similarity = list(zip(points, res))

    sorted_points = [point for point, _ in sorted(points_with_similarity, key=lambda x: x[1], reverse=True)]

    indexes = []
    for p in sorted_points[0:RESULT_SIZE]:
        indexes.append(embeddings.index(p))
    return indexes


def procesar_datos(pdf_file, texto):
    if pdf_file is not None:
        st.success(f"Archivo PDF recibido: {pdf_file.name}")
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pdf_text = '\n'.join([page.extract_text() for page in pdf_reader.pages])
            pdf_text = clean_text(pdf_text)
            st.text_area("Texto extraído del PDF", pdf_text)
        except Exception as e:
            st.exception(e)
    else:
        st.warning("No se subió ningún archivo PDF.")
    
    if texto:
        st.success(f"Texto ingresado: {texto}")
    else:
        st.warning("No se ha escrito texto.")

    try:
        embeddings , chunks = embedding.get_embedding_and_chunks(pdf_text, texto)
        embeddings = embeddings.tolist()
        chunks.append(texto)
        indexes = calculate_nearest_point(embeddings)
        for index in indexes:
            st.text(chunks[index])
        draw_plot(embeddings, chunks, indexes)
    except Exception as e:
        st.exception(e)

st.title("Embedding Visualizer")

pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

texto = st.text_area("Introduce a text")

if st.button("Enviar"):
    procesar_datos(pdf_file, texto)