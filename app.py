# from flask import Flask, request, render_template
# from search_model import search_book_tfidf, search_book_w2v, search_book_fuzzy
#
# app = Flask(__name__)
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/search', methods=['POST'])
# def search():
#     query = request.form['query']
#
#     # Set the similarity threshold
#     tfidf_threshold = 0.1
#     w2v_threshold = 0.1
#     fuzzy_threshold = 80
#
#     results_tfidf = search_book_tfidf(query, threshold=tfidf_threshold)
#     results_w2v = search_book_w2v(query, threshold=w2v_threshold)
#     results_fuzzy = search_book_fuzzy(query, threshold=fuzzy_threshold)
#
#     return render_template('results.html', query=query, results_tfidf=results_tfidf, results_w2v=results_w2v,
#                            results_fuzzy=results_fuzzy)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template
from search_model import search_book_combined
import re

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Kiểm tra nếu từ khóa chứa ký tự số
    if re.search(r'\d', query):
        message = "Không tìm thấy sách"
        results_combined = []
    else:
        # Đặt ngưỡng tối thiểu cho điểm số
        threshold = 0.4
        results_combined = search_book_combined(query, threshold=threshold)

        # Kiểm tra xem có kết quả tìm kiếm hay không
        if not results_combined:
            message = "Không tìm thấy sách"
        else:
            message = None

    return render_template('results.html', query=query, results_combined=results_combined, message=message)


if __name__ == "__main__":
    app.run(debug=True)
