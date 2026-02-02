# XÂY DỰNG WEBSITE THƯƠNG MẠI ĐIỆN TỬ SÁCH TÍCH HỢP HỆ THỐNG GỢI Ý SẢN PHẨM DỰA TRÊN THUẬT TOÁN COLLABORATIVE FILTERING

## Mục đích

Dự án này dựa trên [source website bán sách](https://github.com/rd003/BookShoppingCart-Mvc) của [Ravindra Devrani](https://github.com/rd003) và chúng tôi đã tích hợp thêm hệ thống gợi ý sản phẩm dựa trên thuật toán Collaborative Filtering bằng cách dùng python như một dịch vụ bên ngoài!

## Yêu cầu

- .NET Core 6.0
- Visual Studio 2022
- Python 3.12
- Visual Studio Code

Hoặc bạn có thể dùng docker để chạy dự án này

## Hướng dẫn cài đặt

1. Clone repository này

```bash
git clone https://github.com/Glasspham/AI-GTVTHCM.git
```

2. Cd vào thư mục `Bài tập lớn`

```bash
cd BTL/source_code
```

#### Chạy bằng local

3. Cài đặt các package trong file requirements.txt

```bash
pip install -r requirements.txt
```

4. Cài đặt các package trong file .csproj

```bash
dotnet restore
```

5. Chạy dự án

```bash
dotnet run
```

6. Chạy server python

```bash
uvicorn main:app --reload
```

#### Chạy bằng docker

3. Chạy dự án bằng docker

```bash
docker-compose up -d --build
```

## Hướng dẫn sử dụng

1. Truy cập vào `http://localhost:5000`

2. Đăng nhập

```bash
username: admin@gmail.com
password: Admin@123
```