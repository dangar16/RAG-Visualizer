from sklearn.decomposition import PCA
from dotenv import load_dotenv
import numpy as np
import requests
import os

load_dotenv()

class Embeddings:
    def __init__(self):
        self.api_key = os.getenv("hf_token")
        self.api_model = os.getenv("model_id")
        self.pca = PCA(n_components=3)
        
        if not self.api_key or not self.api_model:
            raise EnvironmentError("Las variables de entorno 'hf_token' y/o 'model_id' no están configuradas.")

        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.api_model}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_embeddings(self, text):
        if not text:
            raise ValueError("El texto de entrada no puede estar vacío.")

        try:
            response = requests.post(self.api_url, headers=self.headers, json={"inputs": text, "options": {"wait_for_model": True}})
            response.raise_for_status()
           
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_embedding_and_chunks(self, pdf_text, texto, chunk_size, chunk_overlap):
        """
        Args:
            pdf_text (str): Texto extraído de un PDF.
        """
        chunks = self.get_chunks(pdf_text, chunk_size, chunk_overlap)
        embeddings = [self.get_embeddings(chunk) for chunk in chunks]
        embeddings.append(self.get_embeddings(texto))
        array = np.array(embeddings)

        pca_result = self.pca.fit_transform(array)
        return pca_result, chunks
    
    def get_chunks(self, pdf_text, chunk_size, overlap):
        """
        Divide el texto extraído de un PDF en fragmentos más pequeños llamados chunks.

        Args:
            pdf_text (str): Texto extraído de un PDF.
            chunk_size (int): Tamaño de los fragmentos de texto.
            overlap (int): Número de caracteres que se superponen entre fragmentos.
        
        Returns:
            Iterator: Generador de chunks.
        """
        chunks = []
        start = 0
        while start < len(pdf_text):
            end = min(start + chunk_size, len(pdf_text))
            chunk = pdf_text[start:end]

            if len(chunks) >= 1:
                chunk = chunks[-1][-overlap:] + chunk
            
            chunks.append(chunk)
            start = end
        return chunks
        
