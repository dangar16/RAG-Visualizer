import re

def cosine_similarity(embeddings, result_size):
    points = embeddings[0:len(embeddings)-1]
    text = embeddings[-1]

    score = []
    for point in points:
        dot_product = sum(a*b for a, b in zip(point, text))

        magnitude_point = sum(a**2 for a in point) ** 0.5
        magnitude_text = sum(a**2 for a in text) ** 0.5

        cosine_similarity = dot_product / (magnitude_point * magnitude_text)
        score.append(cosine_similarity)
    
    points_with_index = list(enumerate(points))
    sorted_points = sorted(points_with_index, key=lambda x: score[x[0]], reverse=True)
    indexes = [i for i, _ in sorted_points[:result_size]]
    return indexes

def clean_text(pdf_text):
    pdf_text = pdf_text.replace('\n', ' ')
    pdf_text = re.sub(r'\s+', ' ', pdf_text)
    pdf_text = pdf_text.strip()

    pdf_text = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜ\s.,;:¡¿]+', ' ', pdf_text)
    return pdf_text