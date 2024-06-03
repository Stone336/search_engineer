import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Đọc dữ liệu từ file CSV
data = pd.read_csv('E:/DoAnAPI/myapp/cleaned_data/preprocessed_dataset.csv')

# Tải danh sách các từ dừng từ thư viện NLTK
stop_words = set(stopwords.words('english'))

# Hàm tiền xử lý để loại bỏ các từ dừng và chuyển đổi văn bản thành chữ thường
def preprocess_text(text):
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Chuyển đổi văn bản thành chữ thường
    text = text.lower()
    # Tách từ
    tokens = word_tokenize(text)
    # Loại bỏ từ dừng
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Kết hợp các từ lại thành một chuỗi
    processed_text = ' '.join(filtered_tokens)
    return processed_text

# Áp dụng hàm preprocess_text cho mỗi phần tử trong cột "title"
data['title'] = data['title'].apply(preprocess_text)

# Hiển thị dữ liệu đã tiền xử lý
print(data.head())

# Lưu DataFrame đã được tiền xử lý vào file CSV mới
# data.to_csv('E:/DoAnAPI/myapp/cleaned_data/preprocessed_dataset.csv', index=False)
