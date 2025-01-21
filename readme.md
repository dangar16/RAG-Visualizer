# Embedding Visualizer

This application visualizes embeddings for better understanding and analysis.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/dangar16/embeddingVisualizer.git
    cd embeddingVisualizer
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**
    ```bash
    streamlit run main.py
    ```

2. **Open your browser and navigate to:**
    ```
    http://localhost:8501
    ```

## Information
I have used a free open source embedding model sentence-transformers/all-MiniLM-L6-v2 that is available in hugging-face.

This app is made to understand better how RAG works by splitting the test in chunks. Convert each chunk into an embedding to visualize them in a 3D scatter plot.

The sentence-transformers/all-MiniLM-L6-v2 model outputs an embedding with 384 dimensions. In order to visualize the embeddings, I reduced the dimensions using Principal Component Analysis (PCA).

PCA is a dimensionality reduction method used to reduce the dimensionality of large data sets. PCA helps to visualize high-dimensional data by transforming it into a lower-dimensional space. In this case, to a 3D plot.

The 3D plot contains black dots that represent the chunks, a red dot that represents the text written, and green dots that are the 'x' most similar chunks compared to the text.

To compare each chunk with the text, I've used cosine similarity to measure the distance between the points.


## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.