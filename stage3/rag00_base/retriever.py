

import numpy as np

# 相似度、关键词、混合检索


# 余弦相似度
def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)

def cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
    document_norm = normalize_rows(documents)
    return document_norm @ query_norm

def test_cosine_similarity():
    vectors = np.array([
            [0.8, 0.1, 0.0],
            [0.0, 0.1, 0.9],
        ], dtype=np.float32)
    query_vector = np.array([0.7, 0.2, 0.0], dtype=np.float32)

    scores = cosine_similarity(query_vector, vectors)
    print(f"scores: {scores}")

