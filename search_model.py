# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import Word2Vec
# from nltk.tokenize import word_tokenize
# import numpy as np
# from fuzzywuzzy import process
#
# # Đọc dữ liệu từ file CSV
# df = pd.read_csv('E:/DoAnAPI/myapp/cleaned_data/cleaned_dataset.csv')
#
# # Lấy cột title
# titles = df['title'].tolist()
#
# # --------------------- TF-IDF + Cosine Similarity ---------------------
#
# # Tạo đối tượng TF-IDF Vectorizer
# tfidf_vectorizer = TfidfVectorizer()
#
# # Chuyển đổi cột title thành ma trận TF-IDF
# tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
#
#
# # Hàm tìm kiếm sách theo tên sử dụng TF-IDF và Cosine Similarity
# def search_book_tfidf(query, threshold=0.1):
#     query_vec = tfidf_vectorizer.transform([query])
#     cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
#     related_docs_indices = [i for i in cosine_similarities.argsort()[:-6:-1] if
#                             cosine_similarities[i] >= threshold]  # Lấy top 5 kết quả với ngưỡng
#     results = [(titles[i], cosine_similarities[i]) for i in related_docs_indices]
#     return results
#
#
# # --------------------- Word2Vec + Cosine Similarity ---------------------
#
# # Tiền xử lý văn bản
# def preprocess(text):
#     return word_tokenize(text.lower())
#
#
# # Tạo tập dữ liệu huấn luyện cho Word2Vec
# sentences = [preprocess(title) for title in titles]
#
# # Huấn luyện mô hình Word2Vec
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
#
#
# # Hàm tính trung bình vector cho một câu
# def avg_feature_vector(sentence, model, num_features):
#     words = preprocess(sentence)
#     feature_vec = np.zeros((num_features,), dtype="float32")
#     n_words = 0
#
#     for word in words:
#         if word in model.wv:
#             n_words += 1
#             feature_vec = np.add(feature_vec, model.wv[word])
#
#     if n_words > 0:
#         feature_vec = np.divide(feature_vec, n_words)
#
#     return feature_vec
#
#
# # Tính vector cho tất cả các tiêu đề sách
# title_vectors = np.array([avg_feature_vector(title, model, 100) for title in titles])
#
#
# # Tìm kiếm sách với Word2Vec và Cosine Similarity
# def search_book_w2v(query, threshold=0.1):
#     query_vec = avg_feature_vector(query, model, 100)
#     cosine_similarities = cosine_similarity([query_vec], title_vectors).flatten()
#     related_docs_indices = [i for i in cosine_similarities.argsort()[:-6:-1] if
#                             cosine_similarities[i] >= threshold]  # Lấy top 5 kết quả với ngưỡng
#     results = [(titles[i], cosine_similarities[i]) for i in related_docs_indices]
#     return results
#
#
# # --------------------- Fuzzy Search ---------------------
#
# # Hàm tìm kiếm từ gần đúng sử dụng fuzzywuzzy
# def search_book_fuzzy(query, threshold=80):
#     results = process.extract(query, titles, limit=5)
#     results = [(title, score) for title, score in results if score >= threshold]  # Lọc kết quả với ngưỡng
#     return results

# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MinMaxScaler
# import gensim.downloader as api
# from fuzzywuzzy import fuzz
#
# # Load pre-trained Word2Vec model
# w2v_model = api.load('glove-wiki-gigaword-100')
#
# # Load titles from CSV
# df = pd.read_csv('E:/DoAnAPI/myapp/cleaned_data/cleaned_dataset.csv')
# titles = df['title'].tolist()
#
#
# def search_book_tfidf(query):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
#     query_vector = tfidf_vectorizer.transform([query])
#     cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#     results = [(titles[i], score) for i, score in enumerate(cosine_similarities)]
#     return results
#
#
# def search_book_w2v(query):
#     def get_w2v_vector(text):
#         words = text.split()
#         word_vecs = [w2v_model[word] for word in words if word in w2v_model]
#         if word_vecs:
#             return np.mean(word_vecs, axis=0)
#         else:
#             return np.zeros(w2v_model.vector_size)
#
#     query_vector = get_w2v_vector(query)
#     title_vectors = np.array([get_w2v_vector(title) for title in titles])
#     cosine_similarities = cosine_similarity([query_vector], title_vectors).flatten()
#     results = [(titles[i], score) for i, score in enumerate(cosine_similarities)]
#     return results
#
#
# def search_book_fuzzy(query):
#     results = [(title, fuzz.ratio(query.lower(), title.lower())) for title in titles]
#     return results
#
#
# def search_book_combined(query, weights=(0.4, 0.4, 0.4), threshold=0.5):
#     results_tfidf = search_book_tfidf(query)
#     results_w2v = search_book_w2v(query)
#     results_fuzzy = search_book_fuzzy(query)
#
#     # Normalize scores
#     scaler = MinMaxScaler()
#     tfidf_scores = scaler.fit_transform(np.array([score for _, score in results_tfidf]).reshape(-1, 1)).flatten()
#     w2v_scores = scaler.fit_transform(np.array([score for _, score in results_w2v]).reshape(-1, 1)).flatten()
#     fuzzy_scores = scaler.fit_transform(np.array([score for _, score in results_fuzzy]).reshape(-1, 1)).flatten()
#
#     # Combined scores
#     combined_scores = weights[0] * tfidf_scores + weights[1] * w2v_scores + weights[2] * fuzzy_scores
#
#     combined_results = list(enumerate(combined_scores))
#     combined_results = [(i, score) for i, score in combined_results if score >= threshold]  # Filter by threshold
#     combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)[:10]  # Get top 10 results
#
#     results = [(titles[i], score) for i, score in combined_results]
#     return results

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import gensim.downloader as api
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords  # Thêm thư viện nltk

# Load pre-trained Word2Vec model
w2v_model = api.load('glove-wiki-gigaword-100')

# Load titles from CSV
df = pd.read_csv('E:/DoAnAPI/myapp/cleaned_data/cleaned_dataset.csv')
titles = df['title'].tolist()

# Loại bỏ từ dừng tiếng Anh
stop_words = set(stopwords.words('english'))

def search_book_tfidf(query):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # Sử dụng 'english' thay vì stop_words=stop_words
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    results = [(titles[i], score) for i, score in enumerate(cosine_similarities)]
    return results


def search_book_w2v(query):
    def get_w2v_vector(text):
        words = text.split()
        # Loại bỏ từ dừng trong danh sách từ
        words = [word for word in words if word not in stop_words]
        word_vecs = [w2v_model[word] for word in words if word in w2v_model]
        if word_vecs:
            return np.mean(word_vecs, axis=0)
        else:
            return np.zeros(w2v_model.vector_size)

    query_vector = get_w2v_vector(query)
    title_vectors = np.array([get_w2v_vector(title) for title in titles])
    cosine_similarities = cosine_similarity([query_vector], title_vectors).flatten()
    results = [(titles[i], score) for i, score in enumerate(cosine_similarities)]
    return results


def search_book_fuzzy(query):
    results = [(title, fuzz.ratio(query.lower(), title.lower())) for title in titles]
    return results


def search_book_combined(query, weights=(0.4, 0.4, 0.4), threshold=0.5):
    results_tfidf = search_book_tfidf(query)
    results_w2v = search_book_w2v(query)
    results_fuzzy = search_book_fuzzy(query)

    # Normalize scores
    scaler = MinMaxScaler()
    tfidf_scores = scaler.fit_transform(np.array([score for _, score in results_tfidf]).reshape(-1, 1)).flatten()
    w2v_scores = scaler.fit_transform(np.array([score for _, score in results_w2v]).reshape(-1, 1)).flatten()
    fuzzy_scores = scaler.fit_transform(np.array([score for _, score in results_fuzzy]).reshape(-1, 1)).flatten()

    # Combined scores
    combined_scores = weights[0] * tfidf_scores + weights[1] * w2v_scores + weights[2] * fuzzy_scores

    combined_results = list(enumerate(combined_scores))
    combined_results = [(i, score) for i, score in combined_results if score >= threshold]  # Filter by threshold
    combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)[:15]  # Get top 10 results

    results = [(titles[i], score) for i, score in combined_results]
    return results
