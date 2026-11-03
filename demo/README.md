# CSPRL — Research project page

Trang giới thiệu công trình theo cấu trúc học thuật, ưu tiên đầu ra: tên bài và tác giả → kế hoạch và kết quả chính → hình đối chiếu thu gọn → tóm tắt → bảng kết quả và cách đọc chỉ số → GIF/luồng chạy → phương pháp và thông số → giới hạn. Nền trắng, chữ xám, link xanh lam; không có dashboard, bộ lọc, timeline hoặc số liệu giả lập dạng KPI.

Phần diễn giải gồm 7 chỉ số kết quả (định nghĩa, chiều tốt/xấu, đơn vị và đánh đổi), đầu ra của 5 macro-action, 10 đặc trưng nút theo nhóm và 13 mục cấu hình mã hóa/huấn luyện. Các số liệu tổng hợp được giữ nguyên từ bài báo; các chênh lệch như 7,6 điểm R và 7,9 điểm phần trăm ngân sách được tính từ các trung bình báo cáo.

Ảnh kết quả có chiều rộng tối đa 210 px mỗi hình trên desktop; teaser tối đa 720 px. Người đọc có thể mở ảnh gốc qua link của hình. Bảng III/IV được hiển thị trước phần phương pháp để dễ đối chiếu kết quả mà không cần đọc hết cấu hình.

## Xem trang

Từ thư mục gốc repo:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory demo
```

Mở <http://localhost:8765>. Không cần npm hoặc môi trường huấn luyện. Trang cũng đọc được khi mở trực tiếp `demo/index.html`, vì dữ liệu nghiên cứu và các bảng đã nằm trong HTML.

Trang chỉ có một nút dừng/phát ảnh động. GIF được dựng sẵn, tự phát khi người dùng không bật reduced motion. Khi tắt JavaScript hoặc bật giảm chuyển động, poster tĩnh vẫn truyền đạt nội dung và các bảng/hình vẫn hiển thị đầy đủ. Không nạp mô hình, không gọi API, không tính mô phỏng trong trình duyệt.

## Nguồn dữ liệu và hình ảnh

- Năm ảnh bản đồ mới từ `res/` được sao chép **nguyên vẹn** vào `demo/assets/figures/`. Giữ nguyên độ phân giải và màu; bấm hình để mở bản gốc.
- Các file gồm hạ tầng hiện hữu, Greedy-Benefit, Greedy-Demand, GA và RL_102353. Tên file không cung cấp đầy đủ metadata để xác nhận từng ảnh tương ứng chính xác với những lần chạy trong bảng tổng hợp.
- Bảng III/IV và sơ đồ kiến trúc: từ `VNICT_0804_camera_ready.pdf` do người dùng cung cấp. Bảng số liệu là kết quả bài báo, không suy ra từ GIF hoặc ảnh trong res.
- GIF `planning-process.gif`: minh họa bằng quy tắc xác định trên đồ thị Đống Đa hiện tại (670 nút), khác snapshot thực nghiệm 639 nút. Năm trạm khởi tạo và toàn bộ hành động là mẫu; không phải replay của checkpoint DQN.
- Màu GIF: xám là trạm mẫu ban đầu, đỏ là trạm bổ sung, vòng xanh là vị trí đang cập nhật. Mô phỏng chỉ chạy lúc dựng asset, không chạy khi mở trang.
- Đường bộ: © OpenStreetMap contributors. PDF và hình thuộc công trình nguồn; attribution không mặc nhiên cấp quyền tái sử dụng.

## Cập nhật ảnh và GIF

Cần Node.js, Matplotlib và Pillow cho bước dựng tài nguyên; trang đã có sẵn mọi asset nên không cần cài các thư viện này để xem.

```bash
python demo/scripts/render_animation.py
```

Lệnh sao chép năm ảnh mới nhất từ res và dựng lại GIF/poster. Chỉnh chuỗi minh họa tại `demo/simulation.mjs`; đây là quy tắc giả lập, không phải chính sách học được.

Để xuất lại nền đường và hình cắt từ PDF gốc (cần Pillow và pdftoppm):

```bash
python demo/scripts/prepare_assets.py /path/to/VNICT_0804_camera_ready.pdf
python demo/scripts/render_animation.py
```

Cách trình bày tham khảo DeepVis, ExampleNet và Awesome Scientific Figure: bài viết khoa học với hình lớn, caption, phương pháp và dẫn nguồn; không sao chép code hoặc nhận diện thương hiệu.

## Kiểm tra

```bash
node --test demo/tests/simulation.test.mjs
CHROMEDRIVER=/path/to/chromedriver python demo/tests/browser_smoke.py
```

Browser test cần server đang chạy, Selenium và Chrome/ChromeDriver tương thích. Kiểm tra asset, pause/play, reduced motion, nội dung khi không có JavaScript, các bảng tĩnh, màn hình nhỏ và đảm bảo không tải mô hình hoặc mô phỏng phía trình duyệt. Screenshot kiểm tra được lưu tại /tmp.
