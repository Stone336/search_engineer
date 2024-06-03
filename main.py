from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Đọc dữ liệu từ file CSV
df = pd.read_csv('E:/DoAnAPI/myapp/cleaned_data/cleaned_dataset.csv')

# Xây dựng mô hình tìm kiếm
vectorizer = TfidfVectorizer()
X_new = vectorizer.fit_transform([x.lower() for x in df['title']])
df['score'] = np.log(df['average_rating'] * df['ratings_count'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X_new).flatten()
    match_idx = np.where(similarity != 0)[0]

    if len(match_idx) == 0:
        # Nếu không tìm thấy kết quả, hiển thị thông báo
        message = "No results found."
        return render_template('results.html', message=message)
    else:
        indices = np.argsort(-similarity[match_idx])
        correct_indices = match_idx[indices]
        result = df.iloc[correct_indices]

        result['overall'] = result['score'] * similarity[correct_indices]
        result = result.sort_values(by='overall', ascending=False)

        return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)