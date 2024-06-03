import pandas as pd

# Đọc dữ liệu từ CSV vào DataFrame
data = pd.read_csv('E:/DoAnAPI/myapp/cleaned_data/preprocessed_dataset.csv')

# Kiểm tra và xử lý giá trị thiếu trong cột "title"
missing_values = data["title"].isnull().sum()
if missing_values > 0:
    print(f"Số lượng giá trị thiếu trong cột 'title': {missing_values}")
    # # Xử lý giá trị thiếu tại đây (ví dụ: điền giá trị mặc định, loại bỏ các hàng chứa giá trị thiếu, ...)
    # # Ví dụ: điền giá trị mặc định là "Unknown"
    # data.loc[data["title"].isnull(), "title"] = "Unknown"
    # Hoặc loại bỏ các hàng chứa giá trị thiếu
    data.dropna(subset=["title"], inplace=True)

    missing_values = data.isnull().any()
    print("Có giá trị thiếu trong tập dữ liệu sau khi xử lý:", missing_values.any())


# data.to_csv('E:/DoAnAPI/myapp/cleaned_data/preprocessed_dataset.csv', index=False)

