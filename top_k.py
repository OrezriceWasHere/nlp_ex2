import numpy as np
import math


def similarty(u, v) -> float:
    return np.dot(u, v) / math.sqrt(np.dot(u, u) * np.dot(v, v))


def most_similar(word, k) -> np.ndarray:
    word_vec = vec_to_word[word]
    similarities = np.array([similarty(word_vec, vec) for vec in vecs])
    return (np.argsort(similarities)[::-1][:k + 1])[1:]


if __name__ == "__main__":
    vecs = np.loadtxt("data/embedding/wordVectors.txt", dtype=float)
    words = np.loadtxt("data/embedding/vocab.txt", dtype=str)
    words = np.array([word.lower().rstrip() for word in words])
    vec_to_word = {word: tuple(vec) for word, vec in zip(words, vecs)}
    interesting_words = [
        "dog",
        "england",
        "john",
        "explode",
        "office"
    ]
    for word in interesting_words:
        print(f"Most similar words to {word}:")
        for similar_word in most_similar(word, 5):
            print(words[similar_word])
        print()
